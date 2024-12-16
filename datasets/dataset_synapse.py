import os
import h5py
import numpy as np
from torch.utils.data import Dataset


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split="train", transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(
            os.path.join(list_dir, self.split + ".txt"), encoding="utf-8"
        ).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip("\n")
            data_path = os.path.join(self.data_dir, slice_name + ".npz")
            data = np.load(data_path)
            image, label = data["image"], data["label"]
        else:
            vol_name = self.sample_list[idx].strip("\n")
            filepath = self.data_dir + f"/{vol_name}.npy.h5"
            data = h5py.File(filepath)
            image, label = data["image"][:], data["label"][:]

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)

        sample["case_name"] = self.sample_list[idx].strip("\n")
        return sample
