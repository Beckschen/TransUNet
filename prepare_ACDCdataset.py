# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:26:22 2021

@author: Yusra Shahid

This script can be used to prepare ADCDC Dataset for training with TransUNet

"""
import numpy as np
import nibabel as nb
import glob
import matplotlib.pyplot as plt
import cv2
import h5py
import os

## the directory with datatset
root_dir = "../ACDC dataset\\training\*\*"

files = glob.glob(root_dir)
labels = []
images = []

## this code separates the ground truth files from the images
for each in files:
    if "frame" in each and "gt" in each:
        labels.append(each)
    elif "frame" in each:
        images.append(each)
        
## read images and labels and save them as npz file
os.mkdir('../ACDC dataset\\train_npz',exist_ok = True)
prev_patient = "patient001"
slice_num = 0
for i in range(len(images)):
    slice_num=0
    patient = images[i].split("\\")[-2]
    print(patient)
    print(prev_patient)
    image = nb.load(images[i]).get_fdata()
    label = nb.load(labels[i]).get_fdata()
    slices = image.shape[2]
    if i!=0 and prev_patient == patient:
        slice_num = slice_num +slices
    print(slices,slice_num)
    for num in range(slices):
        # resizing using cv2 so the image isn't changed or tiled as with numpy
        case_image = cv2.resize(image[:,:,num],(512,512))
        case_label = cv2.resize(label[:,:,num],(512,512))
       # case['image'] = case_image
       # case['label'] = case_label
        np.savez("../ACDC dataset\\train_npz\\" + str(patient) + "_slice" + str(slice_num).zfill(3),image = case_image, label=case_label)
        slice_num+=1
    prev_patient = patient
    



