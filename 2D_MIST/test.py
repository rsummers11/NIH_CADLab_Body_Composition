import os
import sys
import logging
from tqdm import tqdm
import argparse
from pathlib import Path
from loader import recursive_glob
from unet_res_mixed import resunet_mixed
import torch.nn as nn
import torch
import json
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
import nibabel as nib

def load_data_statistics(file_name):
    res_dict = None
    try:
        f = open(file_name, 'rb')
    except FileNotFoundError:
        print(f"File {file_name} not found.  Aborting")
        sys.exit(1)
    except OSError:
        print(f"OS error occurred trying to open {file_name}")
        sys.exit(1)
    except Exception as err:
        print(f"Unexpected error opening {file_name} is", repr(err))
        sys.exit(1)
    else:
        with open(file_name) as data_file:
            res_dict = json.loads(data_file.read())
    return res_dict

def load_sitk_volume(ctfile):
    sample = {}
    sitk_t1 = sitk.ReadImage(ctfile)
    ct = sitk.GetArrayFromImage(sitk_t1).astype(np.float32)
    direction = sitk_t1.GetDirection()
    spacing = sitk_t1.GetSpacing()
    origin = sitk_t1.GetOrigin()

    sample['data'] = np.transpose(ct, axes=[2, 1, 0])
    sample['direction'] = direction
    sample['spacing'] = spacing
    sample['origin'] = origin

    sample['name'] = os.path.basename(ctfile)[:-7]
    return sample

def normalize_volume(input_vol, cliplow, cliphigh, order=3):
    img_shape = input_vol.shape
    vol = torch.from_numpy(input_vol)
    ratio = (256/img_shape[0], 256/img_shape[1], 1)
    vol = scipy.ndimage.zoom(vol, ratio, order=order)
    vol = np.clip(vol, cliplow, cliphigh)
    img_min = np.min(vol)
    img_max = np.max(vol)
    vol = (vol - img_min) / (img_max - img_min)
    return vol, ratio

def save_volme(file_name, sample, vol):
    if 'affine' in sample.keys():
        seg_res_img = nib.Nifti1Image(vol, sample['affine'], sample['header'])
        nib.save(seg_res_img, file_name)
    else:
        vol = np.transpose(vol, axes=[2, 1, 0])
        vol = sitk.GetImageFromArray(vol)
        vol.SetDirection(sample['direction'])
        vol.SetSpacing(sample['spacing'])
        vol.SetOrigin(sample['origin'])
        sitk.WriteImage(vol, file_name)

def normalize_volume1(input_vol, stat_dict, order=3):
    img_shape = input_vol.shape
    vol = torch.from_numpy(input_vol)
    ratio = (256/img_shape[0], 256/img_shape[1], 1)
    vol = scipy.ndimage.zoom(vol, ratio, order=order)

    vol = np.clip(vol, stat_dict['percentile_00_5'], stat_dict['percentile_99_5'])
    vol = (vol - stat_dict['mean']) / max(stat_dict['std'], 1e-8)
    return vol, ratio

def segment_fat_muscle_dual_branch(ct_file_name, output_dir, seg_model,
                                   data_statistics_dict):

    sample = load_sitk_volume(ct_file_name)
    if data_statistics_dict is None:
        vol, ratio = normalize_volume(input_vol=sample['data'],
                                      cliplow=-1000, cliphigh=800,
                                      order=3)
    else:
        vol, ratio = normalize_volume1(input_vol=sample['data'],
                                       stat_dict=data_statistics_dict,
                                       order=3)
    # Extract fat regions
    seg_res_strong = np.zeros(vol.shape).astype(int)
    seg_res_weak = np.zeros(vol.shape).astype(int)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        pbar = tqdm(desc='Segmenting CT slices', total=vol.shape[2], ncols=100)
        for j in range(vol.shape[2]):
            # print('Segmenting {}-th slice of {} slices for the CT scan of {}\n'.format(
            #     j, vol.shape[2], sample['name']))
            slice = vol[:, :, j]
            slice = torch.from_numpy(slice)
            slice = torch.round(slice * 1000) / 1000  # round to make more compressible
            slice = torch.unsqueeze(slice, 0)
            slice = torch.unsqueeze(slice, 0)
            slice = slice.to(device)
            result = seg_model(slice)

            # if isinstance(result, list):
            #     result = result[0]
            result[0] = torch.softmax(result[0], dim=1)
            result[0] = torch.argmax(result[0], dim=1)
            result[0] = torch.squeeze(result[0])
            result[0] = result[0].cpu().numpy()
            seg_res_strong[:, :, j] = result[0]

            result[1] = torch.softmax(result[1], dim=1)
            result[1] = torch.argmax(result[1], dim=1)
            result[1] = torch.squeeze(result[1])
            result[1] = result[1].cpu().numpy()
            seg_res_weak[:, :, j] = result[1]
            pbar.update(1)
        pbar.close()

    seg_res_strong = scipy.ndimage.zoom(seg_res_strong, (1.0 / ratio[0], 1.0 / ratio[1],
                                             1 / ratio[2]), order=0)
    seg_res_strong = seg_res_strong.astype(np.uint8)
    file_path = os.path.join(output_dir, sample['name'] + "_strong.nii.gz")
    save_volme(file_path, sample, seg_res_strong)

    seg_res_weak = scipy.ndimage.zoom(seg_res_weak, (1.0 / ratio[0], 1.0 / ratio[1],
                                                         1 / ratio[2]), order=0)
    seg_res_weak = seg_res_weak.astype(np.uint8)
    file_path = os.path.join(output_dir, sample['name'] + "_weak.nii.gz")
    save_volme(file_path, sample, seg_res_weak)

class WrappedModel(nn.Module):
	def __init__(self, module):
		super(WrappedModel, self).__init__()
		self.module = module # that I actually define.
	def forward(self, x):
		return self.module(x)

def create_fat_seg_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seg_model = resunet_mixed(in_class=1, out_class=4)
    seg_model = seg_model.to(device)
    print(seg_model)
    checkpoint = torch.load(model_path, map_location=device)

    if torch.cuda.is_available():
        seg_model = nn.DataParallel(seg_model)
    else:
        seg_model = WrappedModel(seg_model)

    try:
        if 'state_dict' in checkpoint:
            seg_model.load_state_dict(checkpoint['state_dict'])
        elif 'network_weights' in checkpoint:
            seg_model.load_state_dict(checkpoint['network_weights'])
        else:
            seg_model.load_state_dict(checkpoint)
    except RuntimeError:
        seg_model = resunet_mixed(in_class=1, out_class=4)
        seg_model = seg_model.to(device)
        # try one more time..
        try:
            if 'state_dict' in checkpoint:
                seg_model.load_state_dict(checkpoint['state_dict'])
            elif 'network_weights' in checkpoint:
                seg_model.load_state_dict(checkpoint['network_weights'])
            else:
                seg_model.load_state_dict(checkpoint)
        except RuntimeError:
            seg_model = None
            print('cannot be assigned with the model file {}'.format(model_path))
            return seg_model

    # main_frame.set_text(isinstance(seg_model.module.final_activation, nn.Softmax))
    print("=> loaded checkpoint '{}')".format(model_path))

    seg_model.train(False)
    seg_model.eval()
    return seg_model

def inference_data(input_file_list, model_weights_file, output_dir):
    if not os.path.isfile(model_weights_file) or not os.path.exists(model_weights_file):
        print('The model weights file {} is not existed'.format(model_weights_file))
        return

    if not os.path.isdir(output_dir) or not os.path.exists(output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    seg_model = create_fat_seg_model(model_weights_file)
    if seg_model is None:
        print('The segmentation process is terminated because the model cannot be assigned with the weights')
        return

    data_statistics_file = os.path.join(os.path.dirname(model_weights_file),
                                        'training_data_statistics.txt')
    data_statistics_dict = load_data_statistics(data_statistics_file)

    iterator = tqdm(range(len(input_file_list)), ncols=70)
    for f_id in iterator:
        print('\n Segmenting {}'.format(input_file_list[f_id]))
        segment_fat_muscle_dual_branch(input_file_list[f_id], output_dir, seg_model,
                                       data_statistics_dict)

def main():
    # ------------------------------- argument parsing --------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input_dir', type=str,
                        help='Input directory containing testing data in nii.gz format', default=None)
    parser.add_argument('-i', '--input_file_list', type=str,
                        help='Input file list in nii.gz format', default=None)
    parser.add_argument('-m', '--model_file', type=str,
                        help='The file with trained model weights',
                        required=True)
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Output directory that saves segmentation results',
                        required=True)
    args = parser.parse_args()

    if args.input_dir is not None:
        image_list = recursive_glob(args.input_dir, '.nii.gz')
    elif args.input_file_list is not None:
        # Read input filelist
        with open(args.input_file_list) as f:
            image_list = f.read().splitlines()
    else:
        print('No input files are found')
        return

    inference_data(image_list, args.model_file, args.output_dir)

if __name__ == '__main__':
    main()