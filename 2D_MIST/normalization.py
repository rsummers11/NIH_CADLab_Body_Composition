import os
import numpy as np
from loader import recursive_glob
import nibabel as nib
from abc import ABC, abstractmethod
from typing import Type
from numpy import number

def compute_image_statistics(img, label=None):
    if label is None:
        intensity_statistics_res = {
            'mean': float(np.mean(img)),
            'median': float(np.median(img)),
            'std': float(np.std(img)),
            'min': float(np.min(img)),
            'max': float(np.max(img)),
            'percentile_99_5': float(np.percentile(img, 99.5)),
            'percentile_00_5': float(np.percentile(img, 0.5)),
        }
    else:
        foreground_pixels = img[label>0]
        intensity_statistics_res = {
            'mean': float(np.mean(foreground_pixels)),
            'median': float(np.median(foreground_pixels)),
            'std': float(np.std(foreground_pixels)),
            'min': float(np.min(foreground_pixels)),
            'max': float(np.max(foreground_pixels)),
            'percentile_99_5': float(np.percentile(foreground_pixels, 99.5)),
            'percentile_00_5': float(np.percentile(foreground_pixels, 0.5)),
        }

    return intensity_statistics_res

def compute_dataset_statistics(img_dir, label_dir=None):
    img_files = recursive_glob(img_dir, '.nii.gz')
    img_files.sort()
    imgs_statistics_arr = {
        'mean_arr': [],
        'median_arr': [],
        'std_arr': [],
        'min_arr': [],
        'max_arr': [],
        'percentile_99_5_arr': [],
        'percentile_00_5_arr': [],
    }

    if label_dir is None or len(img_files) != len(recursive_glob(label_dir, '.nii.gz')):
        for f in img_files:
            img = nib.load(f).get_fdata()
            img_statistics = compute_image_statistics(img)
            imgs_statistics_arr['mean_arr'].append(img_statistics['mean'])
            imgs_statistics_arr['median_arr'].append(img_statistics['median'])
            imgs_statistics_arr['std_arr'].append(img_statistics['std'])
            imgs_statistics_arr['min_arr'].append(img_statistics['min'])
            imgs_statistics_arr['max_arr'].append(img_statistics['max'])
            imgs_statistics_arr['percentile_99_5_arr'].append(img_statistics['percentile_99_5'])
            imgs_statistics_arr['percentile_00_5_arr'].append(img_statistics['percentile_00_5'])
    else:
        label_files = recursive_glob(label_dir, '.nii.gz')
        label_files.sort()
        for f1, f2 in zip(img_files, label_files):
            img = nib.load(f1).get_fdata()
            label = nib.load(f2).get_fdata()
            img_statistics = compute_image_statistics(img, label)
            imgs_statistics_arr['mean_arr'].append(img_statistics['mean'])
            imgs_statistics_arr['median_arr'].append(img_statistics['median'])
            imgs_statistics_arr['std_arr'].append(img_statistics['std'])
            imgs_statistics_arr['min_arr'].append(img_statistics['min'])
            imgs_statistics_arr['max_arr'].append(img_statistics['max'])
            imgs_statistics_arr['percentile_99_5_arr'].append(img_statistics['percentile_99_5'])
            imgs_statistics_arr['percentile_00_5_arr'].append(img_statistics['percentile_00_5'])

    return {
        'mean': float(np.mean(imgs_statistics_arr['mean_arr'])),
        'median': float(np.median(imgs_statistics_arr['median_arr'])),
        'std': float(np.std(imgs_statistics_arr['mean_arr'])),
        'min': float(np.median(imgs_statistics_arr['min_arr'])),
        'max': float(np.median(imgs_statistics_arr['max_arr'])),
        'percentile_99_5': float(np.median(imgs_statistics_arr['percentile_99_5_arr'])),
        'percentile_00_5': float(np.median(imgs_statistics_arr['percentile_00_5_arr'])),
    }

class ImageNormalization(ABC):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = None

    def __init__(self, use_mask_for_norm: bool = None, intensityproperties: dict = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        assert isinstance(intensityproperties, dict)
        self.intensityproperties = intensityproperties
        self.target_dtype = target_dtype

    @abstractmethod
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass

class ZScoreNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = True

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        image = image.astype(self.target_dtype)
        if self.use_mask_for_norm is not None and self.use_mask_for_norm:
            # negative values in the segmentation encode the 'outside' region (think zero values around the brain as
            # in BraTS). We want to run the normalization only in the brain region, so we need to mask the image.
            # The default nnU-net sets use_mask_for_norm to True if cropping to the nonzero region substantially
            # reduced the image size.
            mask = seg >= 0
            mean = image[mask].mean()
            std = image[mask].std()
            image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        else:
            mean = image.mean()
            std = image.std()
            image = (image - mean) / (max(std, 1e-8))
        return image

class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image

class NoNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image.astype(self.target_dtype)


class RescaleTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        image = image.astype(self.target_dtype)
        image = image - image.min()
        image = image / np.clip(image.max(), a_min=1e-8, a_max=None)
        return image


class RGBTo01Normalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert image.min() >= 0, "RGB images are uint 8, for whatever reason I found pixel values smaller than 0. " \
                                 "Your images do not seem to be RGB images"
        assert image.max() <= 255, "RGB images are uint 8, for whatever reason I found pixel values greater than 255" \
                                   ". Your images do not seem to be RGB images"
        image = image.astype(self.target_dtype)
        image = image / 255.
        return image