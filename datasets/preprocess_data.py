import os
import time
import argparse
from glob import glob

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                   default='../data/Abdomen/RawData', help='download path for Synapse data')
parser.add_argument('--dst_path', type=str,
                   default='../data/Synapse', help='root dir for data')
parser.add_argument('--use_normalize', action='store_true', default=True,
                   help='use normalize')
args = parser.parse_args()


def preprocess_train_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/train_npz", exist_ok=True)
    
    a_min, a_max = -125, 275
    
    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][3:7]
        
        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()
        
        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)
        
        H, W, D = image_data.shape
        
        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))
        
        for dep in range(D):
            save_path = f"{args.dst_path}/train_npz/case{number}_slice{dep:03d}.npz"
            np.savez(save_path, label=label_data[dep,:,:], image=image_data[dep,:,:])
    pbar.close()


def preprocess_valid_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/test_vol_h5", exist_ok=True)
    
    a_min, a_max = -125, 275

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][3:7]
        
        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()
        
        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)
        
        H, W, D = image_data.shape
        
        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))
        
        save_path = f"{args.dst_path}/test_vol_h5/case{number}.npy.h5"
        f = h5py.File(save_path, 'w')
        f['image'] = image_data
        f['label'] = label_data
        f.close()
    pbar.close()


if __name__ == "__main__":
    data_root = f"{args.src_path}/Training"
    
    # String sort
    image_files = sorted(glob(f"{data_root}/img/*.nii.gz"))
    label_files = sorted(glob(f"{data_root}/label/*.nii.gz"))
    
    preprocess_train_image(image_files, label_files)
    preprocess_valid_image(image_files, label_files)
