from __future__ import print_function, division
import os
import torch
import numpy as np
import pdb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from random import random, randint
import nibabel as nib
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot,filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def load_segmentation_model(model_dir, model_file_name):
    os.makedirs(model_dir, exist_ok=True)
    model_file_list = recursive_glob(model_dir, '.pth')
    model_file_list.sort(reverse=True)
    for f in model_file_list:
        if model_file_name in f:
            model_file_name = os.path.basename(f)
            break

    return model_file_name

#---------------------------------------------------------------------------------------
def load_checkpoint(filename, model, device):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    # start_epoch = 0
    # if os.path.isfile(filename):
    #     print("=> loading checkpoint '{}'".format(filename))
    #     if torch.cuda.is_available():
    #         checkpoint = torch.load(filename)
    #     else:
    #         checkpoint = torch.load(filename, map_location = torch.device('cpu'))
    #
    #     # checkpoint = torch.load(filename, map_location = torch.device('cpu'))
    #     start_epoch = checkpoint['epoch']
    #     best_miou = checkpoint.get('miou',0)
    #
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(filename, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(filename))
    #
    # return model, start_epoch, best_miou
    if isinstance(filename, str):
        checkpoint = torch.load(filename, map_location=device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
    new_state_dict = {}
    for k, value in checkpoint['network_weights'].items():
        key = k
        if key not in model.state_dict().keys() and key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value

    return checkpoint, new_state_dict

def save_checkpoint(model_file_name, current_epoch, model, optimizer, logger, best_miou):
    checkpoint = {
        'network_weights': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'logging': logger.get_checkpoint(),
        '_best_ema': best_miou,
        'current_epoch': current_epoch,
    }
    torch.save(checkpoint, model_file_name)

def make_dataset(root):
    items = []

    train_img_path = os.path.join(root, 'images')
    train_mask_path = os.path.join(root, 'masks')

    # images = os.listdir(train_img_path)
    # labels = os.listdir(train_mask_path)
    images = recursive_glob(train_img_path, '.nii.gz')
    labels = recursive_glob(train_mask_path, '.nii.gz')

    images.sort()
    labels.sort()

    for it_im, it_gt in zip(images, labels):
        # item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
        item = (it_im, it_gt)
        items.append(item)


    return items

class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, up_sample_size=-1, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = make_dataset(root_dir)
        self.up_sample_size = up_sample_size

    def __len__(self):
        if self.up_sample_size == -1:
            return len(self.imgs)
        else:
            return  self.up_sample_size

    # def augment2(self, img, mask):
    #     if random() > 0.5:
    #         img = ImageOps.flip(img)
    #         mask = ImageOps.flip(mask)
    #     if random() > 0.5:
    #         img = ImageOps.mirror(img)
    #         mask = ImageOps.mirror(mask)
    #     if random() > 0.5:
    #         angle = random() * 60 - 30
    #         img = img.rotate(angle)
    #         mask = mask.rotate(angle)
    #     return img, mask
    #
    # def augment(self, img, mask, mask_w):
    #     if random() > 0.5:
    #         img = ImageOps.flip(img)
    #         mask = ImageOps.flip(mask)
    #         mask_w = ImageOps.flip(mask_w)
    #     if random() > 0.5:
    #         img = ImageOps.mirror(img)
    #         mask = ImageOps.mirror(mask)
    #         mask_w = ImageOps.mirror(mask_w)
    #     if random() > 0.5:
    #         angle = random() * 60 - 30
    #         img = img.rotate(angle)
    #         mask = mask.rotate(angle)
    #         mask_w = mask_w.rotate(mask_w)
    #     return img, mask, mask_w

    def __getitem__(self, index):

        if self.up_sample_size != -1:
            index = int(index * len(self.imgs) / self.up_sample_size)

        img_path, mask_path = self.imgs[index]
        img = nib.load(img_path)
        seg = nib.load(mask_path)
        img = img.get_fdata()
        seg = seg.get_fdata()

        sample = {'img': img, 'seg': seg}

        if self.transform:
            sample = self.transform(sample)

        # return sample, img_path, mask_path
        return sample


class MedicalImageDataset1(MedicalImageDataset):
    """Face Landmarks dataset."""

    def __getitem__(self, index):

        sample = super(MedicalImageDataset1, self).__getitem__(index)
        img_path, mask_path = self.imgs[index]
        return sample, img_path, mask_path