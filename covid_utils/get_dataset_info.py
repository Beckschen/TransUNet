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
def print_dataset_info(): 
    print("数据集A的图像大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(datasetA.shape, datasetA.max(), datasetA.min(), datasetA.dtype))
    print("数据集A的标注大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(masksetA.shape, masksetA.max(), masksetA.min(), masksetA.dtype))
    print("数据集B的图像大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(datasetB.shape, datasetB.max(), datasetB.min(), datasetB.dtype))
    print("数据集B的标注大小为 {}, 数组中最小值为 {}, 最大值为 {}, 数组类型为 {}".format(masksetB.shape, masksetB.max(), masksetB.min(), masksetB.dtype))

def print_dataset_label_info(labelset) :
    """
    不具备通用性的方法，用来统计从kaggle上下载的两个数据集不同类型标注的数量，即毛玻璃和实型
    """
    groundclass_num = consolidation_num = total_num = shared_num = 0 
    for i in tqdm.tqdm(range(0, labelset.shape[0])) :
        is_contain_groundclass = True in labelset[i, :, :, 0]
        is_contain_consolidation = True in labelset[i, :, :, 1]
        if is_contain_groundclass and is_contain_consolidation :
            groundclass_num += 1
            consolidation_num += 1
            total_num += 1
            shared_num += 1
        elif is_contain_groundclass and not is_contain_consolidation :
            groundclass_num += 1
            total_num += 1
        elif not is_contain_groundclass and is_contain_consolidation :
            consolidation_num += 1
            total_num += 1
        elif not is_contain_groundclass and not is_contain_consolidation :
            print("编号为{}的切片不含有任何标注".format(i))
    print("该数据集总计切片数目 {},\n包含部分病灶（毛玻璃或实型）的切片数目 {},\n包含毛玻璃的切片数目 {},\n包含实型的切片数目 {},\n包含全部病灶（毛玻璃和实型）的切片数目 {}".format(i + 1, total_num, groundclass_num, consolidation_num, shared_num))

def test1():
    visual_test_path = "/home/wubo/BiYeExp/TempTest/"
    create_visual(datasetA[20, :, :, 0], masksetA[20, :, :, 0], 1, visual_test_path + "test1.png", 1)
    create_visual_muti_label(datasetA[20, :, :, 0], masksetA[20, :, :], visual_test_path + "label.png", -1)

def create_dataset_visual():
    """
    对数据集进行可视化处理
    """
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

if __name__ == "__main__":
    # print_dataset_info()
    # create_dataset_visual()
    # print_dataset_label_info(masksetA)
    # print_dataset_label_info(masksetB)
    test1()
