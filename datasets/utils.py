import random

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


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
    def __init__(self, output_size, order=3):
        """
        order int, optional
            The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
        """
        self.output_size = output_size
        self.order = order

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=self.order,
            )
            label = zoom(
                label, (self.output_size[0] / x, self.output_size[1] / y), order=0
            )

        assert (image.shape[0] == self.output_size[0]) and (
            image.shape[1] == self.output_size[1]
        )
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {"image": image, "label": label}
        return sample
