import itertools
import os
import re
import h5py
import numpy as np
from torch.utils.data import Dataset


class ACDC_dataset(Dataset):
    def __init__(self, base_dir, list_dir=None, split="train", transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        train_ids, val_ids, test_ids = self._get_ids()
        if self.split.find("train") != -1:
            self.all_slices = os.listdir(self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            for ids in train_ids:
                new_data_list = list(
                    filter(
                        lambda x: re.match("{}.*".format(ids), x) != None,
                        self.all_slices,
                    )
                )
                self.sample_list.extend(new_data_list)

        elif self.split.find("val") != -1:
            self.all_volumes = os.listdir(self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in val_ids:
                new_data_list = list(
                    filter(
                        lambda x: re.match("{}.*".format(ids), x) != None,
                        self.all_volumes,
                    )
                )
                self.sample_list.extend(new_data_list)

        elif self.split.find("test") != -1:
            self.all_volumes = os.listdir(self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(
                    filter(
                        lambda x: re.match("{}.*".format(ids), x) != None,
                        self.all_volumes,
                    )
                )
                self.sample_list.extend(new_data_list)

        print(f"total {len(self.sample_list)} samples.")

    def _get_ids(self):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        testing_set = ["patient{:0>3}".format(i) for i in range(1, 21)]
        validation_set = ["patient{:0>3}".format(i) for i in range(21, 31)]
        training_set = [
            i for i in all_cases_set if i not in testing_set + validation_set
        ]

        return [training_set, validation_set, testing_set]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        if self.split == "train":
            h5f = h5py.File(
                self._base_dir + f"/ACDC_training_slices/{case}", "r"
            )

        else:
            h5f = h5py.File(
                self._base_dir + f"/ACDC_training_volumes/{case}", "r"
            )
        
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}

        if self.split == "train":
            sample = self.transform(sample)
        
        sample["idx"] = idx
        sample["case_name"] = case.replace(".h5", "")
        return sample


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
