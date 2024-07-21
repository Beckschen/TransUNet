import torch
from torchvision import transforms
from PIL import Image
import os
import time
from FLA_TransUNet import VisionTransformer, CONFIGS

# 加载模型配置和权重
config = CONFIGS['ViT-B_16']  # 使用特定的配置
model_path = 'model/FocusedLinearAttention_skip3_epo50_bs24_224.pth.pth'
model = VisionTransformer(config, img_size=224, num_classes=2)

# 加载模型权重
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# 只加载匹配的权重
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # 标准化（更新的均值和标准差适用于灰度图像）
    transforms.Lambda(lambda x: x.expand(3, -1, -1))  # 将单通道图像扩展为三通道
])


def process_image(image_path, output_folder):
    # 开始计时
    start_time = time.time()

    # 打开图片
    image = Image.open(image_path).convert("L")  # 确保图像为灰度模式
    # 预处理图像
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # 添加批次维度

    # 确保模型在不计算梯度的情况下进行推理
    model.eval()  # 切换到推理模式
    with torch.no_grad():
        output = model(input_batch)

    # 获取分割掩码
    seg_mask = torch.argmax(output, dim=1).squeeze(0)  # 获取预测类别，移除批次维度

    # 转换为PIL图像
    seg_mask_pil = Image.fromarray(seg_mask.byte().cpu().numpy() * 255)  # 乘以255以便可视化

    # 获取原始文件名并添加后缀
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    processed_image_name = f"{name}_segmented{ext}"
    processed_image_path = os.path.join(output_folder, processed_image_name)

    # 保存处理后的分割掩码图像
    seg_mask_pil.save(processed_image_path)

    # 结束计时
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time} seconds")

    return processed_image_path
