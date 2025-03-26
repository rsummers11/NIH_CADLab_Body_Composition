import os.path

from loader import MedicalImageDataset, MedicalImageDataset1, recursive_glob, load_checkpoint
from torchvision import transforms
import augmentation
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch
import nibabel as nib

random_seed = 42

def get_train_validate_loader(root_dir, img_statistics, scale_factor, num_classes = 1,
                              batch_size = 4, validation_ratio=0.1,
                              shuffle=True, full_img_num = -1,
                              num_workers = 4, pin_memory=False):
    error_msg = 'validation_ratio {} should be in the range [0, 1]'.format(validation_ratio)
    assert ((validation_ratio >= 0) and (validation_ratio <= 1)), print(error_msg)

    """ prepare train/val set """
    train_set = MedicalImageDataset(root_dir=root_dir, up_sample_size=full_img_num,
                                    transform=transforms.Compose([augmentation.Rescale(scale_factor),
                                                                  augmentation.Rotation(),
                                                                  augmentation.RandomTranslate((20, 20)),
                                                                  augmentation.ElasticTransform(),
                                                                  augmentation.IntensityAdjustmentGlobal(),
                                                                  augmentation.Normalization(img_statistics),
                                                                  augmentation.ToTensor()]))

    val_set = MedicalImageDataset(root_dir=root_dir, up_sample_size=full_img_num,
                                  transform=transforms.Compose([augmentation.Rescale(scale_factor),
                                                                augmentation.Normalization(img_statistics),
                                                                augmentation.ToTensor()]))

    num_train = len(train_set)
    indices = []
    if full_img_num == -1:
        mask_path_list = recursive_glob(os.path.join(root_dir, 'masks'), '.nii.gz')
        mask_path_list.sort()
        for id in range(num_train):
            seg = nib.load(mask_path_list[id])
            seg = seg.get_fdata()
            if int(np.max(seg) + 0.5) == num_classes - 1:
                indices.append(id)
        num_train = len(indices)
    else:
        indices = list(range(num_train))

    split = int(float(validation_ratio * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, num_train

def get_train_loader(root_dir, img_statistics, scale_factor, batch_size = 1,
                     full_img_num = -1, num_workers = 4, pin_memory=False):
    """ prepare train/val set """
    train_set = MedicalImageDataset1(root_dir=root_dir, up_sample_size=full_img_num,
                                     transform=transforms.Compose([augmentation.Rescale(scale_factor),
                                                                   augmentation.Normalization(img_statistics),
                                                                   augmentation.ToTensor()]))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader