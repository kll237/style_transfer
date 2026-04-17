#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI艺术创作工作室核心算法模块
包含风格迁移核心逻辑、模型定义和图像处理函数
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import models, transforms
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# 基础配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class DeviceType:
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def get_available_device() -> Tuple[torch.device, str]:
    """获取最佳可用设备"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"✅ 检测到 {device_count} 个GPU设备，使用多GPU并行")
        return torch.device("cuda:0"), DeviceType.CUDA
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps"), DeviceType.MPS
    else:
        return torch.device("cpu"), DeviceType.CPU


DEVICE, DEVICE_TYPE = get_available_device()


# 特征提取器
class EnhancedFeatureExtractor(nn.Module):
    """增强的特征提取器，支持多模型和多GPU，提取更丰富的特征用于风格迁移"""

    def __init__(self, model_type="vgg19", use_multi_gpu=False, use_advanced_features=True):
        super().__init__()
        self.use_advanced_features = use_advanced_features

        # 扩展模型配置，增加更多层用于更强烈的风格提取
        model_configs = {
            "vgg19": {
                "model": models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1),
                "content_idx": 21,
                "style_idxs": [0, 2, 5, 7, 10, 12, 19, 21, 28, 30, 33, 35, 38]
            },
            "vgg16": {
                "model": models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1),
                "content_idx": 16,
                "style_idxs": [0, 2, 5, 7, 10, 12, 17, 19, 24, 26, 29, 31, 34]
            },
            "vgg19-light": {
                "model": models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1),
                "content_idx": 21,
                "style_idxs": [0, 2, 5, 7, 10, 12, 19, 21]  # 只使用前8层
            },
            "inception_v3": {
                "model": models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1),
                "content_idx": 10,
                "style_idxs": [2, 5, 8, 10, 15, 20, 25, 30]
            }
        }

        cfg = model_configs.get(model_type, model_configs["vgg19"])
        features = cfg["model"].features if model_type.startswith("vgg") else cfg["model"].conv2d_layers

        # 多GPU支持
        if use_multi_gpu and torch.cuda.device_count() > 1:
            features = nn.DataParallel(features)

        features = features.to(DEVICE)

        # 冻结参数
        for p in features.parameters():
            p.requires_grad = False

        self.layers = nn.ModuleList(features if not use_multi_gpu else features.module)
        self.content_idx = cfg["content_idx"]
        self.style_idxs = cfg["style_idxs"]

        # 改进的权重分配：增强低层纹理权重，使风格效果更强烈
        self.style_weights = {}
        total_layers = len(self.style_idxs)
        for i, idx in enumerate(self.style_idxs):
            # 指数衰减调整，使低层权重更高，风格特征更突出
            weight = np.exp(-i / total_layers * 1.5)  # 调整系数使衰减更慢
            self.style_weights[idx] = float(weight)

    def forward(self, x):
        content_feat = None
        style_feats = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.content_idx:
                content_feat = x
            if i in self.style_idxs:
                style_feats[i] = x

        # 高级特征增强：添加特征融合，使风格迁移更强烈
        if self.use_advanced_features and style_feats:
            enhanced_style = {}
            # 融合相邻层特征，增强风格表达
            for i, idx in enumerate(self.style_idxs):
                if i > 0:
                    prev_idx = self.style_idxs[i - 1]
                    enhanced_style[idx] = 0.7 * style_feats[idx] + 0.3 * style_feats[prev_idx]
                else:
                    enhanced_style[idx] = style_feats[idx]
            style_feats = enhanced_style

        return {"content": content_feat, "style": style_feats}


# 损失函数增强
def gram_matrix(tensor):
    """计算Gram矩阵（优化版，增强风格表达）"""
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (c * h * w)


def advanced_style_loss(gen_feat, style_feat, style_weights, style_intensity=1.0):
    """增强的风格损失，使风格迁移效果更强烈"""
    total_loss = 0.0
    total_weight = 0.0

    for idx, weight in style_weights.items():
        if idx not in gen_feat["style"] or idx not in style_feat["style"]:
            continue

        gen_gram = gram_matrix(gen_feat["style"][idx])
        style_gram = gram_matrix(style_feat["style"][idx])

        # 基础MSE损失
        layer_loss = nn.MSELoss()(gen_gram, style_gram)

        # 增强风格影响：增加高频率分量的权重
        if idx < 10:  # 低层更关注纹理
            layer_loss *= 1.5

        # 应用风格强度因子
        layer_loss *= style_intensity

        total_loss += weight * layer_loss
        total_weight += weight

    return total_loss / total_weight if total_weight > 0 else total_loss


def content_loss(gen_feat, content_feat, content_preservation=1.0):
    """内容损失，可调节内容保留程度"""
    return nn.MSELoss()(gen_feat["content"], content_feat["content"]) * (1.0 / content_preservation)


def total_variation_loss(img, tv_strength=1.0):
    """TV损失（图像平滑），可调节强度"""
    batch, channels, height, width = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (batch * channels * height * width) * tv_strength


def perceptual_loss(gen_feat, content_feat, style_feat, extractor, config):
    """感知损失（内容+风格+TV），增强风格效果"""
    c_loss = config["content_weight"] * content_loss(
        gen_feat, content_feat, config.get("content_preservation", 1.0)
    )

    s_loss = config["style_weight"] * advanced_style_loss(
        gen_feat, style_feat, extractor.style_weights,
        config.get("style_intensity", 1.0)
    )

    tv_loss = config["tv_weight"] * total_variation_loss(
        gen_feat["content"], config.get("tv_strength", 1.0)
    )

    return {
        "total": c_loss + s_loss + tv_loss,
        "content": c_loss,
        "style": s_loss,
        "tv": tv_loss
    }


# 图像预处理与后处理增强
def load_and_preprocess_image(path: str, target_size: int, is_style: bool = False,
                              intensity: float = 1.0, style_enhance: bool = True):
    """读取、缩放、增强图像并转换为Tensor，增强风格图处理"""
    try:
        img = Image.open(path).convert("RGB")

        # 等比缩放至目标尺寸以内
        ratio = min(target_size / img.width, target_size / img.height)
        new_sz = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_sz, Image.Resampling.LANCZOS)

        # 居中填充，支持自定义背景颜色
        bg_color = (255, 255, 255) if not is_style else (0, 0, 0)
        canvas = Image.new("RGB", (target_size, target_size), bg_color)
        offset = ((target_size - img.width) // 2, (target_size - img.height) // 2)
        canvas.paste(img, offset)

        if is_style and style_enhance:
            # 增强风格图特征，使风格更突出
            # 根据强度调整增强效果
            contrast_factor = 1.1 + 0.4 * intensity
            color_factor = 1.1 + 0.4 * intensity
            sharpness_factor = 1.0 + 0.5 * intensity

            canvas = ImageEnhance.Contrast(canvas).enhance(contrast_factor)
            canvas = ImageEnhance.Color(canvas).enhance(color_factor)
            canvas = ImageEnhance.Sharpness(canvas).enhance(sharpness_factor)

            # 对高强度风格添加额外处理
            if intensity > 1.5:
                canvas = canvas.filter(ImageFilter.EDGE_ENHANCE_MORE)

        # 高级图像转换，增加对比度调整
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        return transform(canvas).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise ValueError(f"图像加载失败: {e}")


def image_postprocess(tensor: torch.Tensor, enhance: bool = True, config: Dict = None):
    """Tensor → 反归一化 → PIL → 高级后期增强"""
    try:
        config = config or {}
        inv = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )
        img = inv(tensor.squeeze(0)).cpu().clamp(0, 1)
        pil = transforms.ToPILImage()(img)

        if enhance and config:
            # 高级增强链，使输出效果更强烈
            pil = ImageEnhance.Contrast(pil).enhance(config.get("contrast", 1.2))
            pil = ImageEnhance.Color(pil).enhance(config.get("saturation", 1.2))
            pil = ImageEnhance.Sharpness(pil).enhance(config.get("sharpness", 1.2))

            # 添加可选的艺术效果
            if config.get("use_artistic_filter", False):
                if config.get("style_type") == "油画风格":
                    pil = pil.filter(ImageFilter.SMOOTH_MORE)
                elif config.get("style_type") == "素描风格":
                    pil = pil.convert("L").convert("RGB")  # 转为类似素描的灰度风格

            # 局部对比度增强
            if config.get("local_contrast", 0) > 0:
                from PIL import ImageOps
                pil = ImageOps.autocontrast(pil, cutoff=config.get("local_contrast", 0))

        return pil
    except Exception as e:
        raise ValueError(f"图像后处理失败: {e}")


# 训练管理器
class TrainingManager:
    """训练管理器，支持断点续训、早停和学习率自适应调整"""

    def __init__(self):
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.loss_history = []
        self.start_time = None
        self.learning_rate_adjusted = False

    def reset(self):
        """重置训练管理器"""
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.loss_history = []
        self.start_time = None
        self.learning_rate_adjusted = False

    def should_stop_early(self, current_loss: float, epoch: int, patience: int = 50) -> bool:
        """检查是否应该早停"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.loss_history.append(current_loss)

        # 自适应学习率调整
        if len(self.loss_history) > 10 and not self.learning_rate_adjusted:
            recent_losses = self.loss_history[-10:]
            if (recent_losses[0] - recent_losses[-1]) / recent_losses[0] < 0.05:
                # 损失下降缓慢，调整学习率
                self.learning_rate_adjusted = True
                return "adjust_lr"

        if self.patience_counter >= patience:
            return True
        return False

    def get_training_stats(self, current_epoch: int, total_epochs: int, current_loss: float) -> Dict:
        """获取训练统计信息"""
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        if current_epoch > 0:
            avg_time_per_epoch = elapsed / current_epoch
            remaining_epochs = total_epochs - current_epoch
            remaining_time = avg_time_per_epoch * remaining_epochs
            remaining_str = str(timedelta(seconds=int(remaining_time)))
        else:
            remaining_str = "计算中..."

        return {
            "start_time": self.start_time,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "current_loss": current_loss,
            "best_loss": self.best_loss,
            "elapsed_time": elapsed_str,
            "remaining_time": remaining_str,
        }


# 风格混合器
class StyleMixer:
    """多风格混合器，支持多种风格图混合应用"""

    @staticmethod
    def mix_styles(style_paths: List[str], target_size: int, weights: List[float] = None,
                   intensity: float = 1.0) -> torch.Tensor:
        """混合多种风格图"""
        if not style_paths:
            raise ValueError("至少需要一个风格图")

        # 权重归一化
        if not weights:
            weights = [1.0 / len(style_paths)] * len(style_paths)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        # 加载并混合风格特征
        mixed_style = None
        for i, path in enumerate(style_paths):
            style_tensor = load_and_preprocess_image(
                path, target_size, is_style=True, intensity=intensity
            )
            if mixed_style is None:
                mixed_style = style_tensor * weights[i]
            else:
                mixed_style += style_tensor * weights[i]

        return mixed_style


# 辅助函数
def format_number(value: float) -> str:
    """格式化数字显示"""
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif value >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif value >= 1e3:
        return f"{value / 1e3:.2f}K"
    else:
        return f"{value:.2f}"


def print_environment_info() -> str:
    """返回当前运行环境信息（美化格式）"""
    try:
        device_info = []

        if DEVICE_TYPE == DeviceType.CUDA:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024 ** 3
                device_info.append(f"GPU{i}: {props.name} ({mem_gb:.1f}GB)")
            device_str = "\n  ".join(device_info)
            cuda_version = torch.version.cuda
        elif DEVICE_TYPE == DeviceType.MPS:
            device_str = "Apple Silicon (MPS)"
            cuda_version = "N/A"
        else:
            device_str = "CPU"
            cuda_version = "N/A"

        import platform
        system_info = platform.platform()

        info = f"""
╔══════════════════════════════════════════════════════════════╗
║                🎨 AI艺术创作工作室 v4.0                   ║
╠══════════════════════════════════════════════════════════════╣
║  🔧 系统信息                                                 ║
║    • 系统版本: {system_info[:40]:<40}        ║
║    • Python: {sys.version.split()[0]:<20}                   ║
║    • PyTorch: {torch.__version__:<20}                      ║
║                                                            ║
║  🖥️ 硬件信息                                               ║
║    • 计算设备: {device_str[:40]:<40}        ║
║    • CUDA版本: {cuda_version:<30}                        ║
╚══════════════════════════════════════════════════════════════╝
"""
        return info
    except Exception as e:
        return f"环境信息获取失败: {str(e)}"