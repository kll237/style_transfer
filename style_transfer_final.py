import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # 【关键修复】显式导入torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 路径配置（自动获取脚本所在目录）--------------------------
import matplotlib
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CONTENT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "content.jpg")
STYLE_IMAGE_PATH = os.path.join(PROJECT_ROOT, "style.jpg")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 超参数（RTX 2060+CUDA 12.1最优配置）
IMAGE_SIZE = 512
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 1e10
LEARNING_RATE = 0.001
EPOCHS = 200
SAVE_INTERVAL = 50

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"=== 环境信息 ===")
print(f"Python版本：3.8.6 | PyTorch版本：{torch.__version__} | TorchVision版本：{torchvision.__version__}")
print(f"GPU型号：{torch.cuda.get_device_name(0)} | 显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
print(f"依赖库：OpenCV {cv2.__version__} | Matplotlib {matplotlib.__version__} | Pillow {Image.PILLOW_VERSION}")
print(f"项目路径：{PROJECT_ROOT}")
print(f"图像路径：{CONTENT_IMAGE_PATH} | {STYLE_IMAGE_PATH}")
print(f"结果路径：{OUTPUT_DIR}")
print(f"===============\n")

# -------------------------- 图像预处理/后处理 --------------------------
def image_preprocess(image_path, size=IMAGE_SIZE):
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(DEVICE, non_blocking=True)

def image_postprocess(tensor):
    inv_transform = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = inv_transform(tensor.squeeze(0)).cpu()
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)

# -------------------------- VGG19特征提取器 --------------------------
class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE)
        for param in vgg19.parameters():
            param.requires_grad = False
        
        self.content_layer = "conv4_2"
        self.style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        
        self.layer_map = {}
        conv_count = 1
        relu_count = 1
        pool_count = 1
        for idx, layer in enumerate(vgg19):
            if isinstance(layer, nn.Conv2d):
                layer_name = f"conv{conv_count}_{relu_count}"
                self.layer_map[layer_name] = layer
                relu_count += 1
            elif isinstance(layer, nn.ReLU):
                layer_name = f"relu{conv_count}_{relu_count-1}"
                self.layer_map[layer_name] = layer
                if idx < len(vgg19)-1 and isinstance(vgg19[idx+1], nn.MaxPool2d):
                    conv_count += 1
                    relu_count = 1
            elif isinstance(layer, nn.MaxPool2d):
                layer_name = f"pool{pool_count}"
                self.layer_map[layer_name] = layer
                pool_count += 1
        
        self.layers = nn.ModuleList(self.layer_map.values())
        self.layer_names = list(self.layer_map.keys())
    
    def forward(self, x):
        features = {}
        for name, layer in zip(self.layer_names, self.layers):
            x = layer(x)
            if name == self.content_layer or name in self.style_layers:
                features[name] = x
            if name == self.style_layers[-1]:
                break
        return features

# -------------------------- 损失函数 --------------------------
def content_loss(gen_features, content_features):
    return nn.MSELoss()(gen_features["conv4_2"], content_features["conv4_2"])

def gram_matrix(tensor):
    batch_size, channels, height, width = tensor.size()
    features = tensor.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)

def style_loss(gen_features, style_features):
    total_loss = 0.0
    for layer in style_features.keys():
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = gram_matrix(style_features[layer])
        total_loss += nn.MSELoss()(gen_gram, style_gram)
    return total_loss / len(style_features)

def total_loss(gen_features, content_features, style_features):
    return CONTENT_WEIGHT * content_loss(gen_features, content_features) + STYLE_WEIGHT * style_loss(gen_features, style_features)

# -------------------------- 核心训练流程 --------------------------
def main():
    # 加载图像
    try:
        if not os.path.exists(CONTENT_IMAGE_PATH):
            raise FileNotFoundError(f"未找到内容图：{CONTENT_IMAGE_PATH}\n请将内容图命名为content.jpg，放在项目文件夹")
        if not os.path.exists(STYLE_IMAGE_PATH):
            raise FileNotFoundError(f"未找到风格图：{STYLE_IMAGE_PATH}\n请将风格图命名为style.jpg，放在项目文件夹")
        
        content_tensor = image_preprocess(CONTENT_IMAGE_PATH)
        style_tensor = image_preprocess(STYLE_IMAGE_PATH)
        print(f"✅ 图像加载成功！尺寸：{IMAGE_SIZE}×{IMAGE_SIZE}")
    except Exception as e:
        print(f"❌ 图像加载失败！")
        print(f"错误详情：{e}")
        return
    
    # 初始化组件
    gen_tensor = content_tensor.clone().requires_grad_(True)
    feature_extractor = VGG19FeatureExtractor()
    content_features = feature_extractor(content_tensor)
    style_features = feature_extractor(style_tensor)
    optimizer = optim.Adam([gen_tensor], lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    print(f"✅ VGG19特征提取器加载完成")
    print(f"✅ 特征提取完成（内容层：conv4_2 | 风格层：conv1_1~conv5_1）")
    print(f"\n🚀 开始风格迁移训练（{EPOCHS}轮迭代）...")
    
    # 迭代训练
    for epoch in tqdm(range(EPOCHS), desc="训练进度", ncols=100):
        gen_features = feature_extractor(gen_tensor)
        loss = total_loss(gen_features, content_features, style_features)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 保存中间结果
        if (epoch + 1) % SAVE_INTERVAL == 0:
            intermediate_img = image_postprocess(gen_tensor.detach())
            intermediate_path = os.path.join(OUTPUT_DIR, f"intermediate_epoch_{epoch+1}.jpg")
            intermediate_img.save(intermediate_path)
            print(f"\n📌 Epoch [{epoch+1}/{EPOCHS}] | 总损失：{loss.item():.2f} | 中间结果：{intermediate_path}")
    
    # 保存最终结果和对比图
    final_img = image_postprocess(gen_tensor.detach())
    final_path = os.path.join(OUTPUT_DIR, "final_result.jpg")
    final_img.save(final_path)
    
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(Image.open(CONTENT_IMAGE_PATH).resize((IMAGE_SIZE, IMAGE_SIZE)))
    plt.title("Content Image", fontsize=14)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(Image.open(STYLE_IMAGE_PATH).resize((IMAGE_SIZE, IMAGE_SIZE)))
    plt.title("Style Image", fontsize=14)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_img)
    plt.title("Style Transferred Image", fontsize=14)
    plt.axis("off")
    
    comparison_path = os.path.join(OUTPUT_DIR, "comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print(f"\n🎉 训练完成！所有结果保存在：{OUTPUT_DIR}")
    print(f"1. 最终风格迁移图：{final_path}")
    print(f"2. 中间迭代结果：intermediate_epoch_50/100/150/200.jpg")
    print(f"3. 三图对比图：{comparison_path}")

if __name__ == "__main__":
    main()