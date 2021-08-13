from pathlib import Path

import argparse
import h5py
import nibabel
import numpy


def id_to_color(id: float) -> numpy.ndarray:
    class_to_color_map = {
        "background": "#000000",
        "dimgray": "#696969",
        "lightgray": "#d3d3d3",
        "forestgreen": "#228b22",
        "darkred": "#8b0000",
        "olive": "#808000",
        "lightseagreen": "#20b2aa",
        "darkblue": "#00008b",
        "red": "#ff0000",
        "darkorange": "#ff8c00",
        "yellow": "#ffff00",
        "lime": "#00ff00",
        "royalblue": "#4169e1",
        "deepskyblue": "#00bfff",
        "blue": "#0000ff",
        "fuchsia": "#ff00ff",
        "palevioletred": "#db7093",
        "khaki": "#f0e68c",
        "deeppink": "#ff1493",
        "lightsalmon": "#ffa07a",
        "violet": "#ee82ee",
    }
    from PIL import ImageColor
    return numpy.asarray(ImageColor.getrgb(list(class_to_color_map.values())[int(id)]))


# TODO: how to download testdata?
# TODO: specify in README which exact dataset has to be downloaded
def get_case_ids_from_list(dataset_list_path: Path):
    with open(dataset_list_path, "r") as f:
        slices = f.readlines()
    case_ids = sorted(list(set([s.split("_")[0][4:] for s in slices])))
    return case_ids


def main(args: argparse.Namespace):
    image_dir = args.original_dataset_dir / 'img'
    case_ids = get_case_ids_from_list(args.list_path)
    case_ids = ["0001"]  # TODO
    for case_id in case_ids:
        case_image_dir = image_dir / case_id
        if not case_image_dir.exists():
            print(f"Sub-directory {case_image_dir} doesn't seem to exist. Skipping")
            continue

        for image_path in case_image_dir.iterdir():
            label_id = f"label{image_path.name[3:]}"  # cuts "img" from the image filename and replaces it with "label"
            label_path = args.original_dataset_dir / "label" / case_id / label_id
            assert image_path.exists() and label_path.exists(), f'For id {case_id} either the image or label file ' \
                                                                f'is missing'
            image_data = nibabel.load(image_path).get_fdata()
            label_data = nibabel.load(label_path).get_fdata()

            clipped_image_data = numpy.clip(image_data, *args.clip)
            normalised_image_data = (clipped_image_data - args.clip[0]) / (args.clip[1] - args.clip[0])

            # Reorders data so that the channel dimension is at the front for easier indexing later
            transposed_image_data = numpy.transpose(normalised_image_data, (2, 0, 1))
            transposed_label_data = numpy.transpose(label_data, (2, 0, 1))

            # Extracting slices for training
            for i, (image_slice, label_slice) in enumerate(zip(transposed_image_data, transposed_label_data)):
                out_filename = args.target_dataset_dir / f'Synapse/train_npz/case{case_id}_slice{i:03d}.npz'

                # TODO: remove
                tmp_image = numpy.repeat(numpy.expand_dims(image_slice, axis=2), 3, axis=2)
                tmp_label = numpy.zeros((*label_slice.shape, 3), dtype=numpy.uint8)
                # for id in numpy.unique(label_data):
                #     if id == 0.0:
                #         continue
                #     mask = numpy.where(label_slice == id)
                #     tmp_label[mask] = id_to_color(id)
                combined_array = numpy.concatenate(((tmp_image * 255).astype(numpy.uint8), tmp_label), axis=1)
                from PIL import Image
                Image.fromarray(combined_array, mode="RGB").show()

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
    parser.add_argument('-td', '--target-dataset-dir', type=Path, default=Path('../../data'),
                        help='The directory where the processed dataset should be stored.')
    parser.add_argument('-lp', '--list-path', type=Path, default=Path('../lists/lists_Synapse/train.txt'),
                        help='Path to one of the dataset lists that contain the case ids that should be used.')
    parser.add_argument('--clip', nargs=2, type=float, default=[-125, 275],
                        help='Two numbers [min max] that represent the interval that should be clipped from the '
                             'original image data.')
    parsed_args = parser.parse_args()
    main(parsed_args)
