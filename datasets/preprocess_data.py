import argparse
from pathlib import Path
from typing import List

import h5py
import nibabel
import numpy
from tqdm import tqdm


def get_case_ids_from_list(dataset_list_path: Path) -> List[str]:
    with open(dataset_list_path, "r") as f:
        slices = f.readlines()
    case_ids = sorted(list(set([s.split("_")[0][4:].rstrip() for s in slices])))
    return case_ids


def get_case_ids_from_directory(directory: Path) -> List[str]:
    return [f.stem for f in directory.iterdir()]


def main(args: argparse.Namespace):
    image_dir = args.original_dataset_dir / "img"
    if args.from_list_file is not None:
        case_ids = get_case_ids_from_list(args.from_list_file)
    else:
        case_ids = get_case_ids_from_directory(image_dir)
    print(f"Processing case ids: {case_ids}")

    for case_id in tqdm(case_ids):
        case_image_dir = image_dir / case_id
        if not case_image_dir.exists():
            print(f"Sub-directory {case_image_dir} doesn't seem to exist. Skipping")
            continue

        for image_path in tqdm(case_image_dir.iterdir(), desc="Processing case files", leave=False):
            label_id = f"label{image_path.name[3:]}"  # cuts "img" from the image filename and replaces it with "label"
            label_path = args.original_dataset_dir / "label" / case_id / label_id
            assert image_path.exists() and label_path.exists(), f"For id {case_id} either the image or label file " \
                                                                f"is missing"
            image_data = nibabel.load(image_path).get_fdata()
            label_data = nibabel.load(label_path).get_fdata()

            clipped_image_data = numpy.clip(image_data, *args.clip)
            normalised_image_data = (clipped_image_data - args.clip[0]) / (args.clip[1] - args.clip[0])

            # Reorders data so that the channel dimension is at the front for easier iteration in the subsequent
            # for-loop
            transposed_image_data = numpy.transpose(normalised_image_data, (2, 0, 1))
            transposed_label_data = numpy.transpose(label_data, (2, 0, 1))

            # Extracting slices for training
            for i, (image_slice, label_slice) in tqdm(enumerate(zip(transposed_image_data, transposed_label_data)),
                                                      desc="Processing slices", leave=False):
                out_filename = args.target_dataset_dir / f"Synapse/train_npz/case{case_id}_slice{i:03d}.npz"
                if not args.overwrite and out_filename.exists():  # Do not overwrite data unless flag is set
                    continue

                if not out_filename.parent.exists():
                    out_filename.parent.mkdir(exist_ok=True, parents=True)
                numpy.savez(out_filename, image=image_slice, label=label_slice)

            # keep the 3D volume in h5 format for testing cases.
            h5_filename = args.target_dataset_dir / f"Synapse/test_vol_h5/case{case_id}.npy.h5"
            if not args.overwrite and h5_filename.exists():  # Do not overwrite data unless flag is set
                continue
            if not h5_filename.parent.exists():
                h5_filename.parent.mkdir(exist_ok=True, parents=True)
            with h5py.File(h5_filename, "w") as f:
                f.create_dataset("image", data=normalised_image_data)
                f.create_dataset("label", data=label_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original_dataset_dir", type=Path,
                        help="The root directory for the downloaded, original dataset")
    parser.add_argument("-td", "--target-dataset-dir", type=Path, default=Path("../../data"),
                        help="The directory where the processed dataset should be stored.")
    parser.add_argument("-fl", "--from-list-file", type=Path,
                        help="Do not process all directories that are contained in the original dataset directory, "
                             "but use those contained in the passed list file. The data in the list must be "
                             "structured as in the train.txt file located in lists/lists_Synapse.")
    parser.add_argument("--clip", nargs=2, type=float, default=[-125, 275],
                        help="Two numbers [min max] that represent the interval that should be clipped from the "
                             "original image data.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite the data present in the target dataset directory")
    parsed_args = parser.parse_args()
    main(parsed_args)
