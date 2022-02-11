import numpy as np

def modify_lung_window(img, lungwin = [-1200, 600]):
    """
    肺部CT图像，肺窗调节，默认肺窗范围[-1200,600]
    """
    img = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    img[img > 1] = 1
    img[img < 0] = 0
    img = (img * 255).astype(np.uint8)
    return img

def img_normalized(img, method = 1):
    """
    将图像的像素值范围归一化到[0,1]的范围，默认使用最大最小归一化方式
    """
    if method == 1:
        max = np.max(img)
        min = np.min(img)
        img = (img - min) / (max - min)
    img = img.astype(np.float16)
    return img

def create_img(img):
    img = modify_lung_window(img)
    img = img_normalized(img)
    return img

def create_mask(mask):
    """
    将mask不同类型的标注当做一个类型进行处理，并降维
    """
    new_mask = mask[:, :, 0]
    new_mask[mask[:, :, 1] == True] = True
    return new_mask