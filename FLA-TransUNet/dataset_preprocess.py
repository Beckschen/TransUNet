# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nb
import glob
import cv2
import os


def npz(source_path, target_path):
    files = glob.glob(source_path)
    labels = []
    images = []

    # this code separates the ground truth files from the images
    for each in files:
        if "frame" in each and "gt" in each:
            labels.append(each)
        elif "frame" in each:
            images.append(each)

    # read images and labels and save them as npz file
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    prev_patient = "patient001"
    slice_num = 0
    for i in range(len(images)):
        slice_num = 0
        patient = images[i].split("\\")[-2]

        image = nb.load(images[i]).get_fdata()
        label = nb.load(labels[i]).get_fdata()
        slices = image.shape[2]
        if i != 0 and prev_patient == patient:
            slice_num = slice_num + slices

        for num in range(slices):
            # resizing using cv2 so the image isn't changed or tiled as with numpy
            case_image = cv2.resize(image[:, :, num], (256, 256))
            case_label = cv2.resize(label[:, :, num], (256, 256))

            # Adjust label values according to specified conditions
            case_label[(case_label > 0) & (case_label <= 1)] = 1
            case_label[(case_label > 1) & (case_label <= 2)] = 2
            case_label[(case_label > 2) & (case_label <= 3)] = 3

            # np.savez(target_path + "\\" + str(patient) + "_slice" + str(slice_num).zfill(3), image=image[:, :, num], label=label[:, :, num])
            np.savez(target_path + "\\" + str(patient) + "_slice" + str(slice_num).zfill(3), image=case_image, label=case_label)
            slice_num += 1
        prev_patient = patient


# Write filenames from directory to output_file.txt
def write_filenames_to_txt(directory, output_file):
    filenames = os.listdir(directory)

    # Filter out non-file entries in directories (e.g. subdirectories)
    filenames = [f for f in filenames if os.path.isfile(os.path.join(directory, f))]

    # Write the file name to the output file
    with open(output_file, 'w') as f:
        for filename in filenames:
            f.write(filename.split('.')[0] + '\n')

    print("Write filenames to txt: Done!")


if __name__ == "__main__":
    # the directory with datatset
    train_source_dir = "../data/ACDC/training/*/*"
    train_target_dir = '../data/ACDC/train_npz'
    npz(train_source_dir, train_target_dir)

    test_source_dir = "../data/ACDC/testing/*/*"
    test_target_dir = '../data/ACDC/test_npz'
    npz(test_source_dir, test_target_dir)

    # write filenames from train_npz directory to test.txt
    train_npz_directory = "../data/ACDC/train_npz"
    write_filenames_to_txt(train_npz_directory, './lists/lists_ACDC/train.txt')

    # write filenames from train_npz directory to train.txt
    test_npz_directory = "../data/ACDC/test_npz"
    write_filenames_to_txt(test_npz_directory, 'lists/lists_ACDC/test.txt')
