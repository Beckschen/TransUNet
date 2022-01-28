import os.path
import numpy as np

# 数据集路径信息
dataset_root_path = "/home/wubo/Dataset"
seta_data_path = os.path.join(dataset_root_path, "images_medseg.npy")
seta_mask_path = os.path.join(dataset_root_path, "masks_medseg.npy")
setb_data_path = os.path.join(dataset_root_path, "images_radiopedia.npy")
setb_mask_path = os.path.join(dataset_root_path, "masks_radiopedia.npy")

# 获取数据集统计数据
datasetA = np.load(seta_data_path)
masksetA = np.load(seta_mask_path)
datasetB = np.load(setb_data_path)
masksetB = np.load(setb_mask_path)

print("数据集A的图像大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(datasetA.shape, datasetA.max(), datasetA.min(), datasetA.dtype))
print("数据集A的标注大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(masksetA.shape, masksetA.max(), masksetA.min(), masksetA.dtype))
print("数据集B的图像大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(datasetB.shape, datasetB.max(), datasetB.min(), datasetB.dtype))
print("数据集B的标注大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(masksetB.shape, masksetB.max(), masksetB.min(), masksetB.dtype))

# 生成可