import logging
import numpy as np
import os,sys
from tqdm import tqdm
import image_process_utils

# 数据集路径信息
dataset_root_path = "/home/wubo/Dataset"
seta_data_path = os.path.join(dataset_root_path, "images_medseg.npy")
seta_mask_path = os.path.join(dataset_root_path, "masks_medseg.npy")
setb_data_path = os.path.join(dataset_root_path, "images_radiopedia.npy")
setb_mask_path = os.path.join(dataset_root_path, "masks_radiopedia.npy")
restore_path = "/home/wubo/BiYeExp/data"

# 由路径读取为数组
datasetA = np.load(seta_data_path)
masksetA = np.load(seta_mask_path)
datasetB = np.load(setb_data_path)
masksetB = np.load(setb_mask_path)

def main():
    logging.basicConfig(filename= restore_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    imgs = []
    masks = []
    for i in tqdm(range(masksetA.shape[0])):
        mask_slice = image_process_utils.create_mask(masksetA[i, :, :, 0 : 2])
        if True in mask_slice:
            img_slice = image_process_utils.create_img(datasetA[i, :, :, 0])
            imgs.append(img_slice)
            masks.append(mask_slice)
    logging.info("遍历数据集A后有效切片数为{}".format(len(imgs)))
    for j in tqdm(range(masksetB.shape[0])):
        mask_slice = image_process_utils.create_mask(masksetB[j, :, :, 0 : 2])
        if True in mask_slice:
            img_slice = image_process_utils.create_img(datasetB[j, :, :, 0])
            imgs.append(img_slice)
            masks.append(mask_slice)
    logging.info("遍历数据集B后有效切片数为{}".format(len(imgs)))
    imgs = np.array(imgs)
    masks = np.array(masks)
    np.save(os.path.join(restore_path, "imgs.npy"), imgs)
    np.save(os.path.join(restore_path, "masks.npy"), masks)

if __name__ == "__main__":
    main()