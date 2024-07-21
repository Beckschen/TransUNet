# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# import cv2
# import nibabel as nb
# # 读取 .nii.gz 文件
# import numpy as np
#
#
# def read_nii_file(filepath):
#     image = sitk.ReadImage(filepath)
#     image_array = sitk.GetArrayFromImage(image)
#     return image_array
#
# # 显示图像
# def display_image(image_array, title="Image"):
#     plt.imshow(image_array, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
#
# # 文件路径
# img_path = "D:\Software\GitHub\FLA-TransUNet\data\ACDC\\training\patient001\patient001_frame01.nii.gz"
# label_path = "D:\Software\GitHub\FLA-TransUNet\data\ACDC\\training\patient001\patient001_frame01_gt.nii.gz"
# #
# # 读取和显示图像
# image_array = read_nii_file(img_path)
# print(f"Image shape: {image_array.shape}")
# label_array = read_nii_file(label_path)
# print(f"Label shape: {label_array.shape}")
#
#
# image = nb.load(img_path).get_fdata()
# label = nb.load(label_path).get_fdata()
# slices = image.shape[2]
# print(f"Image shape: {image_array.shape}")
# print(f"Label shape: {label_array.shape}")
#
# for num in range(slices):
#     print("image shape: ", image[:, :, num].shape, "label shape: ", label[:, :, num].shape)
#
#     # resizing using cv2 so the image isn't changed or tiled as with numpy
#     case_image = cv2.resize(image[:, :, num], (512, 512))
#     case_label = cv2.resize(label[:, :, num], (512, 512))
#     print(np.unique(label[:, :, num]), np.unique(case_label))
    # Adjust label values according to specified conditions
    # case_label[(case_label > 0) & (case_label <= 1)] = 1
    # case_label[(case_label > 1) & (case_label <= 2)] = 2
    # case_label[(case_label > 2) & (case_label <= 3)] = 3

# 如果图像是3D的，选择一个切片显示
# if len(image_array.shape) == 3:
#     slice_index = image_array.shape[0] // 2  # 中间切片
#     display_image(image_array[slice_index], title=f"Slice {slice_index}")
# else:
#     display_image(image_array, title="Image")

import numpy as np
import matplotlib.pyplot as plt

# 加载 npz 文件
# data = np.load('../data/case0007_slice028.npz')
# data = np.load('../data/ACDC/test_npz/patient101_slice000.npz')
# data = np.load('../data/ACDC/train_npz/patient001_slice003.npz')
data = np.load('../predictions/TU_ACDC224/TU_pretrain_R50-ViT-B_16_FocusedLinearAttention_skip3_20k_bs16_224/patient101_slice005.npz')

# 获取图像和标签
image = data['image']
label = data['label']
prediction = data['prediction']

# 打印形状以确认数据
print(f'Image shape: {image.shape}, Label shape: {label.shape}')
# print(f'Image shape: {image.shape}, Label shape: {label.shape}, Prediction shape: {prediction.shape}')
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Image')

plt.subplot(1, 3, 2)
plt.imshow(label, cmap='gray')
plt.title('Label')
#
# plt.subplot(1, 3, 3)
# plt.imshow(prediction, cmap='gray')
# plt.title('Prediction')

# 显示图像和标签
# plt.figure(figsize=(12, 6))
#
# plt.imshow(image, cmap='gray')  # 假设是灰度图像
#
# # Overlay the segmentation label with transparency
# # plt.imshow(label, cmap='jet', alpha=0.5)
# plt.imshow(prediction, cmap='jet', alpha=0.5)


plt.show()

## h5
# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 加载 h5 文件
# file_path = '../data/ACDC/test_npz/patient001_frame01_slice_0.h5'
# with h5py.File(file_path, 'r') as f:
#     # 假设数据集名称是 'image' 和 'label'
#     image = f['image'][:]
#     label = f['label'][:]
#
# # 打印形状以确认数据
# print(f'Image shape: {image.shape}, Label shape: {label.shape}')
#
# # 显示图像和标签
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.title('Image')
# plt.imshow(image, cmap='gray')  # 假设是灰度图像
#
# plt.subplot(1, 2, 2)
# plt.title('Label')
# plt.imshow(label, cmap='gray')  # 假设是灰度图像
#
# plt.show()