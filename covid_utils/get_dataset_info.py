import os.path,sys,tqdm
sys.path.append("..")
import numpy as np
from utils import create_visual, create_visual_muti_label

# 数据集路径信息
dataset_root_path = "/home/wubo/Dataset"
seta_data_path = os.path.join(dataset_root_path, "images_medseg.npy")
seta_mask_path = os.path.join(dataset_root_path, "masks_medseg.npy")
setb_data_path = os.path.join(dataset_root_path, "images_radiopedia.npy")
setb_mask_path = os.path.join(dataset_root_path, "masks_radiopedia.npy")

# 由路径读取为数组
datasetA = np.load(seta_data_path)
masksetA = np.load(seta_mask_path)
datasetB = np.load(setb_data_path)
masksetB = np.load(setb_mask_path)

# 获取数据集统计数据
def print_dataset_info() : 
    print("数据集A的图像大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(datasetA.shape, datasetA.max(), datasetA.min(), datasetA.dtype))
    print("数据集A的标注大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(masksetA.shape, masksetA.max(), masksetA.min(), masksetA.dtype))
    print("数据集B的图像大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(datasetB.shape, datasetB.max(), datasetB.min(), datasetB.dtype))
    print("数据集B的标注大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(masksetB.shape, masksetB.max(), masksetB.min(), masksetB.dtype))

# 生成可视化图片
def create_dataset_visual(dataset, maskset, path, is_only_positive):
    pass

def test():
    visual_test_path = "/home/wubo/BiYeExp/TempTest/"
    create_visual(datasetB[500, :, :, 0], masksetB[500, :, :, 0], 1, visual_test_path + "test1.png", 1)
    create_visual_muti_label(datasetB[714, :, :, 0], masksetB[714, :, :], visual_test_path + "test.png", -1)
    create_visual_muti_label(datasetA[0, :, :, 0], masksetA[0, :, :], visual_test_path + "test2.png", -1)

if __name__ == "__main__":
    # debug
    # visual_test_path = "/home/wubo/BiYeExp/TempTest/"
    # mask_slice = masksetB[714, :, :, 0]
    # if True in mask_slice:
    #     create_visual_muti_label(datasetB[714, :, :, 0], masksetB[714, :, :], visual_test_path + "positve.png", -1)
    # else:
    #     create_visual_muti_label(datasetB[714, :, :, 0], masksetB[714, :, :], visual_test_path + "negative.png", -1)

    visual_root_path = "/home/wubo/Dataset/new_covid_visual"
    # 1. 生成第一个数据集的可视化图像
    for i in tqdm.tqdm(range(0, datasetA.shape[0])):
        mask_slice = masksetA[i, :, :, 0]
        if True in mask_slice:
            create_visual_muti_label(datasetA[i, :, :, 0], masksetA[i, :, :], "{}/{}/{}/{}.png".format(visual_root_path, "datasetA", "positive", i))
        else:
            create_visual_muti_label(datasetA[i, :, :, 0], masksetA[i, :, :], "{}/{}/{}/{}.png".format(visual_root_path, "datasetA", "negative", i))
    
    # 2. 生成第二个数据集的可视化图像
    for j in tqdm.tqdm(range (0, datasetB.shape[0])):
        mask_slice = masksetB[j, :, :, 0]
        if True in mask_slice:
            create_visual_muti_label(datasetB[j, :, :, 0], masksetB[j, :, :], "{}/{}/{}/{}.png".format(visual_root_path, "datasetB", "positive", j))
        else:
            create_visual_muti_label(datasetB[j, :, :, 0], masksetB[j, :, :], "{}/{}/{}/{}.png".format(visual_root_path, "datasetB", "negative", j))