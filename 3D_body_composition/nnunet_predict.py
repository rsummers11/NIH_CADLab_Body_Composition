import argparse
import os
import shutil
import random
import sys
import uuid
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import SimpleITK as sitk
import torch
from pathlib import Path
import contextlib
import warnings
import pathlib

def recursive_glob(rootdir):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith('.nii') or filename.endswith('.nii.gz')
    ]
    
class DummyFile:
    def write(self, x): pass
    def flush(self): pass

@contextlib.contextmanager
def nostdout(verbose=False):
    if not verbose:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout
    else:
        yield

def load_volume(ctfile):
    sample = {}
    try:
        ctNIB= nib.load(ctfile)
        ct = ctNIB.get_fdata()
        affine = ctNIB.affine

        sample['data'] = ct
        sample['affine'] = affine
        sample['header'] = ctNIB.header

    except:
        sitk_t1 = sitk.ReadImage(ctfile)
        ct = sitk.GetArrayFromImage(sitk_t1)
        direction = sitk_t1.GetDirection()
        spacing = sitk_t1.GetSpacing()
        origin = sitk_t1.GetOrigin()

        sample['data'] = np.transpose(ct, axes=[2, 1, 0])
        sample['direction'] = direction
        sample['spacing'] = spacing
        sample['origin'] = origin

    sample['name'] = os.path.basename(ctfile)[:-7]
    return sample

def search_corresponding_ct_file(seg_f, input_ct_files):
    if len(input_ct_files) == 0:
        raise  ValueError("No input CT files were set!")
    else:
        for ct_f in input_ct_files:
            ct_base_f = os.path.basename(ct_f)
            seg_base_f = os.path.basename(seg_f)[:-7]
            if seg_base_f in ct_base_f:
                return ct_f
        return input_ct_files[0]


def save_segmentation_region_areas(ct_file_name, seg_file_name, output_file_name):

    ct_sample = load_volume(ct_file_name)
    seg_sample = load_volume(seg_file_name)
    if 'header' in ct_sample.keys():
        spacing = ct_sample['header'].get_zooms()
    else:
        spacing = ct_sample['spacing']

    seg_vol = seg_sample['data']
    vol_per_voxel = spacing[0]*spacing[1]*spacing[2]
    res_db = pd.DataFrame(columns=['slice_index', 'muscle volume (mm^3)', 'subcutaneous fat volume (mm^3)',
                                   'visceral fat volume (mm^3)', 'muscle area (mm^2)', 'subcutaneous fat area (mm^2)',
                                   'visceral fat area (mm^2)', 'muscle mean density (HU)', 'muscle density std (HU)',
                                   'subcutaneous fat mean density (HU)', 'subcutaneous fat density std (HU)',
                                   'visceral fat mean density (HU)', 'visceral fat density std (HU)'])


    for idx in range(seg_vol.shape[2]):
        res_record = []
        res_record.append(idx+1)

        slice = seg_vol[:, :, idx]
        ct_slice = ct_sample['data'][:, :, idx]
        seg_bool = np.where(slice == 4, 1, 0)
        muscle_number_of_voxels = np.sum(seg_bool)
        res_record.append(muscle_number_of_voxels * vol_per_voxel)

        seg_bool = np.where(slice == 3, 1, 0)
        sf_number_of_voxels = np.sum(seg_bool)
        res_record.append(sf_number_of_voxels * vol_per_voxel)

        seg_bool = np.where(slice == 2, 1, 0)
        vf_number_of_voxels = np.sum(seg_bool)
        res_record.append(vf_number_of_voxels * vol_per_voxel)

        res_record.append(muscle_number_of_voxels * spacing[0] * spacing[1])
        res_record.append(sf_number_of_voxels * spacing[0] * spacing[1])
        res_record.append(vf_number_of_voxels * spacing[0] * spacing[1])

        seg_bool = np.where(slice == 4, 1, 0)
        ct_muscle_region = ct_slice[seg_bool.nonzero()]
        if muscle_number_of_voxels != 0:
            res_record.append(np.mean(ct_muscle_region))
            res_record.append(np.std(ct_muscle_region))
        else:
            res_record.append(np.nan)
            res_record.append(np.nan)

        seg_bool = np.where(slice == 3, 1, 0)
        ct_sf_region = ct_slice[seg_bool.nonzero()]
        if sf_number_of_voxels != 0:
            res_record.append(np.mean(ct_sf_region))
            res_record.append(np.std(ct_sf_region))
        else:
            res_record.append(np.nan)
            res_record.append(np.nan)

        seg_bool = np.where(slice == 2, 1, 0)
        ct_vf_region = ct_slice[seg_bool.nonzero()]
        if vf_number_of_voxels != 0:
            res_record.append(np.mean(ct_vf_region))
            res_record.append(np.std(ct_vf_region))
        else:
            res_record.append(np.nan)
            res_record.append(np.nan)

        # print(idx, res_record)
        res_db.loc[len(res_db.index)] = res_record

    res_db.to_csv(output_file_name, encoding="utf-8-sig", index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference using nnU-Net predict_from_folder Python API')
    parser.add_argument('-i', '--input_data', help='This program accepts three types of '
                                                   'inputs: 1) a single file in .nii or .nii.gz format; '
                                                   '2) a file fold with all or parts of files in .nii.gz format '
                                                   'its subfolds with these file format also selected; '
                                                   '3) a txt file list a set of .nii or .nii.gz files',
                        required=True)
    parser.add_argument('-o', '----output_data', help='Output directory or file to save MRI segmentation results',
                        required=True)
    parser.add_argument('-m', '--model_directory', help='The directory with 5-fold nnUNet model weights',
                        default='model_weights')
    parser.add_argument('-f', '--fast', help='fast_mode', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', help='Verbose Output', action='store_true', default=False)
    parser.add_argument('-d', '--device', help='Set the device for inference (cuda or cpu)',
                        default='cuda')
    args = vars(parser.parse_args())
    
    # Append 8bit random hex string to ensure tmp_folder is unique
    os.environ['nnUNet_raw'] = args['output_data']
    os.environ['nnUNet_preprocessed'] = args['output_data']
    os.environ['nnUNet_results'] = args['output_data']

    warnings.filterwarnings("ignore", category=UserWarning, module="nnunetv2")
    with nostdout():
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    # The input data is a file fold
    if os.path.isdir(args['input_data']):
        image_list = recursive_glob(args['input_data'])
    else:
        if args['input_data'].endswith('.txt'):
            with open(args['input_data']) as f:
                tmp_image_list = f.read().splitlines()
        else:
            tmp_image_list = []
            tmp_image_list.append(args['input_data'])
        image_list = [f for f in tmp_image_list if os.path.isfile(f) and os.path.exists(f)]

    if len(image_list) != 1 and os.path.isfile(args['output_data']):
        raise  ValueError("invalid output file setup. It should be set as a fold!")

    if os.path.isdir(args['output_data']):
        pathlib.Path(args['output_data']).mkdir(parents=True, exist_ok=True)
    
    print(image_list)
    print("Instantiating nnunet predictor...")
    with nostdout():
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
            perform_everything_on_device=True, device=torch.device(args['device']),
            verbose=args['verbose'], verbose_preprocessing=args['verbose'],
            allow_tqdm=True)

    if os.path.isdir(args['model_directory']):
        model_dir = args['model_directory']
    else:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args['model_directory'])

    print("model_dir: {}".format(model_dir))
    print('output filename: {}'.format(args['output_data']))

    print("Instantiating trained model...")
    with nostdout():
        predictor.initialize_from_trained_model_folder(model_dir,
            use_folds=(0, 1, 2, 3, 4) if args['device'] != 'cpu' or args['fast'] else (0,),
            checkpoint_name='checkpoint_final.pth',
        )

    print("Starting prediction...")
    with nostdout():
        progress_bar = tqdm(total=len(image_list))
        # Iterating through data list of unknown length
        for filename in image_list:
            print('filename is {}'.format(filename))
            if os.path.isdir(args['output_data']):
                print('output dir: {}'.format(args['output_data']))

                predictor.predict_from_files([[filename]], args['output_data'],
                                             save_probabilities=False, overwrite=True,
                                             num_processes_preprocessing=2,
                                             num_processes_segmentation_export=2,
                                             folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
            else:
                predictor.predict_from_files([[filename]],
                                             [args['output_data']],
                                             save_probabilities=False, overwrite=True,
                                             num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                             folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

            progress_bar.update(1)

    print("Cleaning up...")
    for i in ['dataset.json', 'plans.json', 'predict_from_raw_data_args.json']:
        try:
            if os.path.isdir(args['output_data']):
                os.remove(args['output_data'] + '/' + i)
            else:
                os.remove(os.path.dirname(os.path.abspath(args['output_data'])) + '/' + i)
        except Exception as e:
            print(i, e)
            pass
    print("DONE!")

    print("Calculating muscle/adipose tissue volume and intensity measurements...")
    if os.path.isdir(args['output_data']):
        seg_file_list = recursive_glob(args['output_data'])
    else:
        seg_file_list = [args['output_data']]

    with nostdout():
        progress_bar = tqdm(total=len(seg_file_list))
        # Iterating through data list of unknown length
        for seg_f in seg_file_list:
            ct_f = search_corresponding_ct_file(seg_f, image_list)
            output_f = seg_f[:-7] + '.csv'
            save_segmentation_region_areas(ct_f, seg_f, output_f)
            progress_bar.update(1)




