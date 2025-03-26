import os
import numpy as np
import nibabel as nib
import glob
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
import torch

import scipy.ndimage

def rotate(img_list, order_list, rot_angle, rot_plane=(0, 1)):
    """
    img_list : list of images to transform with the same transformation
    order_list : order of interpolation
    rot_angle : rotation angle in degree
    rot_plane : rotation is performed in this plane, tuple
    """

    num_img = len(img_list)

    res_img_list = []
    for i in range(num_img):
        cval = 0 if order_list[i] == 0 else -1024
        res_img = scipy.ndimage.rotate(img_list[i], rot_angle, rot_plane,
                                       reshape=False, order=order_list[i],
                                       mode="constant", cval=cval)
        res_img_list.append(res_img)

    return res_img_list


def elastic_transform(img_list, ctrl_pts, order_list, alpha=15, sigma=3):
    """
    img_list : list of images to transform with the same transformation
    ctrl_pts : positions of control points, N*d
    order_list : order of interpolation
    alpha : scaling factor for the deformation
    sigma : smooting factor

    First a random displacement field (sampled from a gaussian distribution) is created for each control point,
    it's then convolved with a gaussian standard deviation, σ determines the field : very small if σ is large,
        like a completely random field if σ is small,
        looks like elastic deformation with σ the elastic coefficent for values in between.
    Then the field is added to an array of coordinates, which is then mapped to the original image.
    """

    num_img = len(img_list)
    shape = img_list[0].shape

    def make_sparse_field(shape, ctrl_pts):
        field = np.zeros(shape)
        num_ctrl_pts = ctrl_pts.shape[0]
        field_vals = np.random.randn(num_ctrl_pts)

        # field[tuple(ctrl_pts[:, 0]), tuple(ctrl_pts[:, 1]), tuple(ctrl_pts[:, 2])] = field_vals
        field[tuple(ctrl_pts[:, 0]), tuple(ctrl_pts[:, 1])] = field_vals

        """for i in range(num_ctrl_pts):
            xx, yy, zz = ctrl_pts[i]
            field[xx, yy, zz] = field_vals[i]"""

        return field

    # smooth the field
    dx = scipy.ndimage.gaussian_filter(make_sparse_field(shape, ctrl_pts), sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter(make_sparse_field(shape, ctrl_pts), sigma, mode="constant", cval=0) * alpha
    if len(shape) == 2:
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = x + dx, y + dy

    elif len(shape) == 3:
        dz = scipy.ndimage.gaussian_filter(make_sparse_field(shape, ctrl_pts), sigma, mode="constant", cval=0) * alpha
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = x + dx, y + dy, z + dz

    else:
        raise ValueError("can't deform because the image is not either 2D or 3D")

    res_img_list = []
    for i in range(num_img):
        res_img = scipy.ndimage.map_coordinates(img_list[i], indices, order=order_list[i]).reshape(shape)
        res_img_list.append(res_img)

    return res_img_list


# augmentation
class Rescale(object):
    """Rescale the images in a sample

    Args:
        scale
    """

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        # img_deformed = skimage.transform.rescale(img, self.scale, order=3, preserve_range=True)
        # segm_deformed = skimage.transform.rescale(segm, self.scale, order=3, preserve_range=True)
        # cl_deformed = skimage.transform.rescale(cl, self.scale, order=3, preserve_range=True)

        img_deformed = scipy.ndimage.zoom(img, self.scale, order=3)
        seg_deformed = scipy.ndimage.zoom(seg, self.scale, order=0)

        return {'img': img_deformed, 'seg': seg_deformed}

class Rotation(object):
    """Rotate the images in a sample
    This simulates mildly rotated volumes.
    This is a callable class which wraps the function 'rotate'
    Please refer to the arguments of the function 'rotate'

    Args:
        rot_range
        rot_plane
    """

    def __init__(self, rot_range=(-10, 10), rot_plane=(0, 1)):
        assert isinstance(rot_range, (tuple, list))
        assert isinstance(rot_plane, (int, tuple))
        self.rot_range = rot_range
        self.rot_plane = rot_plane

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        img_deformed, seg_deformed = \
            rotate([img, seg], [3, 0],
                   rot_angle=np.random.uniform(self.rot_range[0], self.rot_range[1]),
                   rot_plane=self.rot_plane)

        return {'img': img_deformed, 'seg': seg_deformed}

class ElasticTransform(object):
    """Elastically transform the images in a sample.
    This is a callable class which wraps the function 'elastic_transform'
    Please refer to the arguments of the function 'elastic_transform'

    Args:
        alpha_range
        sigma_range
    """

    def __init__(self, alpha_range=(20, 100), sigma_range=(4, 8)):
        assert isinstance(alpha_range, (tuple, list))
        assert isinstance(sigma_range, (tuple, list))
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        ctrl_pts = np.stack(np.where(seg.astype(bool)), axis=1)
        img_deformed, seg_deformed = \
            elastic_transform([img, seg], ctrl_pts, [3, 0],
                              alpha=np.random.uniform(self.alpha_range[0], self.alpha_range[1]),
                              sigma=np.random.uniform(self.sigma_range[0], self.sigma_range[1]))

        return {'img': img_deformed, 'seg': seg_deformed}

class IntensityAdjustmentGlobal(object):
    """Globally adjust the intensity values of the image in a sample
    Only consider a mild variation because we use the Hounsfield Unit as our input

    Args:
        val_range : value to add to every pixel, (min, max)
    """

    def __init__(self, val_range=(-10, 10)):
        assert isinstance(val_range, (tuple, list))
        self.val_range = val_range

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        new_img = img + np.random.uniform(self.val_range[0], self.val_range[1])

        return {'img': new_img, 'seg': seg}

class IntensityAdjustmentLocal(object):
    """Adjust the intensity values of the image in a sample
    only within the GT small bowels
    This simulates local intensity changes due to the contrast medium.
    Intensity values of small bowels are
    air : around -1000
    low : 0~150
    high (gastrografin) : 300~400

    Args:
        thresh_non_enhanced : (relatively) not enhanced region if below than this
        thresh_enhanced : enhanced region if above than this
        max_change : maximum possible intensity change
    """

    def __init__(self, thresh_non_enhanced=150, thresh_enhanced=300, max_change=300, sigma=(1, 4)):
        self.thresh_non_enhanced = thresh_non_enhanced
        self.thresh_enhanced = thresh_enhanced
        self.max_change = max_change
        assert isinstance(sigma, (tuple, list))
        self.sigma = sigma

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']
        shape = img.shape

        non_enhanced = (img < self.thresh_non_enhanced) & seg.astype(np.bool)
        enhanced = (img > self.thresh_enhanced) & seg.astype(np.bool)

        dval = np.random.uniform(0, self.max_change, shape)
        dval_smooth = scipy.ndimage.gaussian_filter(dval, \
                                                    np.random.uniform(self.sigma[0], self.sigma[1]), \
                                                    mode="constant", cval=0)
        sign = -1 * enhanced.astype(dval.dtype) + non_enhanced.astype(dval.dtype)
        dval = dval_smooth * sign
        dval *= seg.astype(dval.dtype)
        new_img = img + dval

        return {'img': new_img, 'seg': seg}

class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        if random.random() < self.p:
            new_img = np.fliplr(img)
            new_seg = np.fliplr(seg)
            return {'img': new_img, 'seg': new_seg}
        return sample

class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        if random.random() < self.p:
            new_img = np.flipud(img) - np.zeros_like(img)
            new_seg = np.flipud(seg) - np.zeros_like(seg)
            return {'img': new_img, 'seg': new_seg}
        return sample

class RandomTranslate(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']
        assert img.size == seg.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0

        img_deformed = scipy.ndimage.interpolation.shift(img, (x_offset, y_offset), order=3, mode='nearest')
        seg_deformed = scipy.ndimage.interpolation.shift(seg, (x_offset, y_offset), order=0, mode='nearest')

        return {'img': img_deformed,
                'seg': seg_deformed
                }

class Normalization(object):
    """Normalize the intensity values of the image in a sample

    Args:
        int_bound : intensity bound, (min, max)
    """

    def __init__(self, img_statics=None):
        # assert isinstance(int_bound, (tuple, list))
        self.image_statistics = img_statics

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']
        assert self.image_statistics is not None, "Image normalization requires intensity properties"

        # clipping
        img = np.clip(img, self.image_statistics['percentile_00_5'],
                      self.image_statistics['percentile_99_5'])
        # img_min = np.min(img)
        # img_max = np.max(img)

        # img = (img - img_min) / (img_max - img_min)

        img = (img - self.image_statistics['mean']) / max(self.image_statistics['std'], 1e-8)

        return {'img': img, 'seg': seg}

# preprocessing
class ZeroCentering(object):
    """Zero center the intensity values of the image in a sample

    Args:
        set_name
    """

    def __init__(self):
        # self.set_name = set_name
        # self.pixel_mean = 0.5013 if 'mix' in set_name else 0.4981
        self.pixel_mean = 0.5013

    def __call__(self, sample):
        img, seg = sample['img'], sample['seg']

        new_img = img - self.pixel_mean

        return {'img': new_img, 'fseg': seg}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors"""

    def __call__(self, sample):
        # swap axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # img = img.transpose((3, 0, 1, 2))
        # label = label.transpose((3, 0, 1, 2))

        """img, segm, cl = sample['img'], sample['segm'], sample['cl']
        img = np.expand_dims(img, axis=0)
        segm = np.expand_dims(segm, axis=0)
        cl = np.expand_dims(cl, axis=0)
        return {'img': torch.from_numpy(img),
                'segm': torch.from_numpy(segm),
                'cl': torch.from_numpy(cl)}"""

        temp = {}
        for cur_key in list(sample.keys()):
            temp[cur_key] = torch.from_numpy(np.expand_dims(sample[cur_key], axis=0))

        return temp