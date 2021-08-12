from pathlib import Path

import argparse
import h5py
import nibabel
import numpy


# TODO: how to download testdata?
def main(args: argparse.Namespace):
    # Assuming filename are sth. like 'DET0000101_avg.nii' or 'DET0000101_avg_seg.nii'
    filename_stems = set([file.stem.split('_')[0] for file in args.original_dataset_dir.iterdir()])

    for filename_stem in filename_stems:
        case_id = filename_stem[-4:]

        image_path = Path(args.original_dataset_dir / f'{filename_stem}_avg.nii')
        label_path = Path(args.original_dataset_dir / f'{filename_stem}_avg_seg.nii')
        assert image_path.exists() and label_path.exists(), f'For id {filename_stem} either the image or label file ' \
                                                            f'is missing'
        image_data = nibabel.load(image_path).get_fdata()
        label_data = nibabel.load(label_path).get_fdata()

        normalised_image_data = image_data / 255

        # Reorders data so that the channel dimension is at the front for easier indexing later
        transposed_image_data = numpy.transpose(normalised_image_data, (2, 0, 1))
        transposed_label_data = numpy.transpose(label_data, (2, 0, 1))

        # Extracting slices for training
        for i, (image_slice, label_slice) in enumerate(zip(transposed_image_data, transposed_label_data)):
            out_filename = args.target_dataset_dir / f'Synapse/train_npz/case{case_id}_slice{i:03d}.npz'
            if not out_filename.parent.exists():
                out_filename.parent.mkdir(exist_ok=True, parents=True)
            numpy.savez(out_filename, image=image_slice, label=label_slice)

        # keep the 3D volume in h5 format for testing cases.
        # TODO: check if this is correct or if the testdata should be downloaded separately
        h5_filename = args.target_dataset_dir / f'Synapse/test_vol_h5/case{case_id}.npy.h5'
        if not h5_filename.parent.exists():
            h5_filename.parent.mkdir(exist_ok=True, parents=True)
        with h5py.File(h5_filename, 'w') as f:
            f.create_dataset('image', data=normalised_image_data)
            f.create_dataset('label', data=label_data)

    # --------
    # cwd = '/content/drive/My Drive/TransUNet/Training-Testing'
    # data_folder = '/content/drive/My Drive/TransUNet/data'
    # subfolders = os.listdir(cwd + '/' + 'img')  # subfolders will be like ['0062', '0064', ...]
    #
    # # I chose subfolder '0066', but maybe you will want to iterate & combine
    # for subfolder in ['0066']:  # subfolders[1:]:
    #     print(subfolder)
    #     tempwd = cwd + '/' + 'img' + '/' + subfolder
    #     files = os.listdir(tempwd)  # files will be like ['img0032-0066.nii.gz', 'img0036-0066.nii.gz', ...]
    #
    #     # iterate over filenames
    #     for filename in files:
    #         print(filename)
    #         righttext = filename[3:]  # get the part 'xxxx-xxxx.nii.gz'
    #         subject = righttext[:4]
    #         img = nib.load(cwd + '/' + 'img' + '/' + subfolder + '/' + 'img' + righttext)
    #         label_data = nib.load(cwd + '/' + 'label' + '/' + subfolder + '/' + 'label' + righttext)
    #
    #         # Convert them to numpy format,
    #         data = img.get_fdata()
    #         label_data = label_data.get_fdata()
    #
    #         # clip the images within [-125, 275],
    #         data_clipped = np.clip(data, -125, 275)
    #
    #         # normalize each 3D image to [0, 1], and
    #         data_normalised = (data_clipped - (-125)) / (275 - (-125))
    #
    #         # extract 2D slices from 3D volume for training cases while
    #         # e.g. slice 000
    #         for i in range(data_normalised.shape[2]):
    #             formattedi = '{:03d}'.format(i)
    #             slice000 = data_normalised[:, :, i]
    #             label_slice000 = label_data[:, :, i]
    #             np.savez(data_folder + '/Synapse/train_npz/case' + subject + '_slice' + formattedi + '.npz',
    #                      image=slice000,
    #                      label=label_slice000)
    #
    #         # keep the 3D volume in h5 format for testing cases.
    #         fn = data_folder + '/Synapse/test_vol_h5/case' + subject + '.npy.h5'
    #         f = h5py.File(fn, 'w')
    #         dset = f.create_dataset('image', data=data_normalised)
    #         dset = f.create_dataset('label', data=label_data)
    #         f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original_dataset_dir', type=Path,
                        help='The root directory for the downloaded, original dataset')
    parser.add_argument('-td', '--target_dataset_dir', type=Path, default=Path('../../data'),
                        help='The directory where the processed dataset should be stored.')
    parsed_args = parser.parse_args()
    main(parsed_args)
