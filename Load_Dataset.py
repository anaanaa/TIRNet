# -*- coding: utf-8 -*-
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
import clip

from tools.transform import BioMedicalGaussianBlur, PhotoMetricDistortion

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def _center_crop_or_pad(arr, out_h, out_w):
    h, w = arr.shape[:2]
    start_h = max((h - out_h) // 2, 0)
    start_w = max((w - out_w) // 2, 0)
    arr = arr[start_h:start_h + out_h, start_w:start_w + out_w, ...]
    h, w = arr.shape[:2]
    pad_h = max(out_h - h, 0)
    pad_w = max(out_w - w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h > 0 or pad_w > 0:
        if arr.ndim == 3:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        arr = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)
    return arr


def random_zoom(image, label, output_size, zoom_range=(0.9, 1.1)):
    scale = np.random.uniform(zoom_range[0], zoom_range[1])
    if image.ndim == 3:
        image = zoom(image, (scale, scale, 1), order=3)
    else:
        image = zoom(image, (scale, scale), order=3)
    if label.ndim == 3:
        label = zoom(label, (scale, scale, 1), order=0)
    else:
        label = zoom(label, (scale, scale), order=0)
    out_h, out_w = int(output_size[0]), int(output_size[1])
    image = _center_crop_or_pad(image, out_h, out_w)
    label = _center_crop_or_pad(label, out_h, out_w)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text_token, text_mask = sample["image"], sample["label"], sample["text_token"], sample["text_mask"]
        image = np.array(image, dtype=np.uint8)
        label = np.array(label, dtype=np.uint8)

        if random.random() < 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() < 0.5:
            image, label = random_rotate(image, label)
        if random.random() < 0.1:
            image, label = random_zoom(image, label, self.output_size, zoom_range=(0.9, 1.1))

        x, y = image.shape[:2]
        if x != self.output_size[0] or y != self.output_size[1]:
            if image.ndim == 3:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            else:
                image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            if label.ndim == 3:
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1), order=0)
            else:
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        if label.ndim == 3 and label.shape[-1] == 1:
            label = label[..., 0]

        image = F.to_tensor(image.astype(np.uint8))
        label = to_long_tensor(label.astype(np.uint8))
        sample = {'image': image, 'label': label, 'text_token': text_token, 'text_mask': text_mask}
        return sample


class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, text_token, text_mask = sample["image"], sample["label"], sample["text_token"], sample["text_mask"]
        image, label = image.astype(np.uint8), label.astype(np.uint8)
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)       
        # text_mask = torch.Tensor(text_mask)
        # sample = {'image': image, 'label': label, 'text_token': text_token, 'text_mask': text_mask}
        sample['image'] = image
        sample['label'] = label
        return sample


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def correct_dims(*images):
    corr_images = []
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class ImageToImage2D(Dataset):

    def __init__(self, dataset_path: str, task_name: str, row_text: str, joint_transform: Callable = None,
                 one_hot_mask: int = False,
                 image_size: int = 224, data_name='MosMed', token_len=18, config=None, mode="train") -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.mask_list = os.listdir(self.output_path)
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask
        self.rowtext = row_text
        self.task_name = task_name
        self.data_name = data_name
        self.token_len = token_len
        self.config = config
        self.mode = mode
        self.BioMedicalGaussianBlur = BioMedicalGaussianBlur(prob=0.5)
        self.PhotoMetricDistortion = PhotoMetricDistortion()

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        if self.data_name == 'MosMed':
            image_filename = self.images_list[idx]
            mask_filename = image_filename
        else:
            mask_filename = self.mask_list[idx]
            image_filename = mask_filename.replace('mask_', '')
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        try:
            image = cv2.resize(image, (self.image_size, self.image_size))
        except:

            print(os.path.join(self.input_path, image_filename))

        # read mask image
        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask[mask <= 0] = 0
        mask[mask > 0] = 1

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        text = self.rowtext[mask_filename]
        text = text.split('\n')

        with torch.no_grad():
            text_token = clip.tokenize(text, context_length=self.token_len, truncate=True).squeeze()
        text_mask = text_token != 0 
        text_mask = text_mask.int()
        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # sample = (image, mask, text_token, text_mask)
        sample = {'image': image, 'label': mask, 'text_token': text_token, 'text_mask': text_mask, "text": text}
        
        if self.mode=="train":
            sample = self.BioMedicalGaussianBlur.transform(sample)
            sample = self.PhotoMetricDistortion.transform(sample)

        if self.joint_transform:
            sample = self.joint_transform(sample)

        return sample, image_filename
