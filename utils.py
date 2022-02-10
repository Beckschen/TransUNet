import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
from matplotlib import patches

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

def create_visual(img, label, color_code, path, thicnkess = -1):
        """
        作用-生成肺部CT切片及标注的可视化文件
        img-浮点数类型数组
        label-布尔型类型数组，大小与img相同
        color_code-颜色编码，1 红，2 绿，3蓝
        thickness-标注线宽度，-1为填充方式
        path-可视化图片存储的绝对路径
        """
        # 1. 为了可视化结果更清晰，调窗
        lungwin = [-1200,600]
        img = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        img[img > 1] = 1
        img[img < 0] = 0
        img = (img * 255).astype(np.uint8)

        # 2. 图像绘制        
        fig, ax = plt.subplots()
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
        # 3. 标注绘制
        if color_code == 1:
            color = (0, 0, 255)
        elif color_code == 2:
            color = (255, 255, 0)
        elif color_code == 3:
            color = (0, 255, 0)
        else :
            color = (0, 255, 255)
        label = label.astype(np.uint8)
        if np.max(label) != 0 : 
            contour, hierarchy = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(color_img, contours = contour, contourIdx = -1, color = color, thickness = thicnkess)                       
        plt.imshow(color_img)
        plt.axis("off")
        # 去除图像周围的白边
        height, width, channels = color_img.shape
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
        plt.savefig(path, dpi = 300)
        plt.close()

def create_visual_muti_label(img, label, path, thicnkess = -1):
        """
        特异性处理函数，不具备通用性
        作用-生成肺部CT切片及标注的可视化文件
        img-浮点数类型数组
        label-布尔型类型数组，比img高一维度代表不同的标注
        color_code-颜色编码，1 红，2 绿，3蓝
        thickness-标注线宽度，-1为填充方式
        path-可视化图片存储的绝对路径
        """
        # 1. 为了可视化结果更清晰，调窗
        lungwin = [-1200,600]
        img = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
        img[img > 1] = 1
        img[img < 0] = 0
        img = (img * 255).astype(np.uint8)

        # 2. 图像绘制        
        fig, ax = plt.subplots()
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
        # 3. 标注绘制
        for i in range(0,4):
            if i == 0:
                color = (0, 0, 255)
            elif i == 1:
                color = (255, 255, 0)
            elif i == 2:
                color = (0, 255, 0)
            else :
                color = (0, 255, 255)
            label_slice = label[:, :, i]
            label_slice = label_slice.astype(np.uint8)
            if np.max(label_slice) != 0 : 
                contour, hierarchy = cv2.findContours(label_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(color_img, contours = contour, contourIdx = -1, color = color, thickness = thicnkess)                       
        plt.imshow(color_img)
        plt.axis("off")
        # 去除图像周围的白边
        height, width, channels = color_img.shape
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)

        # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
        plt.savefig(path, dpi = 300)
        plt.close()

