import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage import zoom


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


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class CBISDataset(Dataset):
    def __init__(self, base_dir, split, transform=None, list_dir=None):
        self.transform = transform
        self.split = split
        if 'train' in split:
            sub_dir = 'training_data_all_CBIS_256'
        else:
            sub_dir = 'test_data_all_CBIS_256'
        self.img_dir = os.path.join(base_dir, sub_dir, 'img')
        self.mask_dir = os.path.join(base_dir, sub_dir, 'msk')
        self.image_list = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.mask_dir, img_name)
        image = np.array(Image.open(img_path).convert('L'))
        label = np.array(Image.open(label_path).convert('L'))
        label = (label > 0).astype(np.uint8)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = os.path.splitext(img_name)[0]
        return sample
