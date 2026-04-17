#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI艺术创作工作室 - Tkinter图形界面
支持多种预设风格和自定义风格图片
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import numpy as np
from datetime import datetime
import threading

# ======================== 全局配置 ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🎯 使用设备: {DEVICE}")
if DEVICE_TYPE == "cuda":
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"📊 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 预设风格配置 - 使用实际存在的图片
PRESET_STYLES = {
    "油画": {"folder": "jjfg", "files": ["微信图片_20260417211304_2997_9.png"]},
    "梵高风格": {"folder": "yhfg", "files": ["y1.png", "y2.png", "y3.png"]},
    "漫画": {"folder": "mhfg", "files": ["m1.png", "m2.png"]},
    "水彩": {"folder": "scfg", "files": ["s1.png", "s2.png", "s3.png", "s4.png"]},
    "素描": {"folder": "smfg", "files": ["s1.png", "s2.png"]},
    "赛博朋克": {"folder": "sbpk", "files": ["cb1.png", "sb2.png"]},
    "扁平插画": {"folder": "cxfg", "files": ["c1.png", "c2.png"]},
    "艺术特效": {"folder": "yt", "files": ["yt1.png", "yt2.png"]},
    "印象派": {"folder": "yxfg", "files": ["yx1.png", "yx2.png"]},
}

# ======================== 核心功能 ========================
def load_image(image_path, image_size=512):
    """加载并预处理图像"""
    import torchvision.transforms as transforms
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    scale = image_size / max(original_size)
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    left = (new_size[0] - image_size) / 2
    top = (new_size[1] - image_size) / 2
    img = img.crop((left, top, left + image_size, top + image_size))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(DEVICE)

def denormalize(tensor):
    """反归一化"""
    mean = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor([1/0.229, 1/0.224, 1/0.225]).view(1, 3, 1, 1).to(DEVICE)
    img = tensor * std + mean
    return torch.clamp(img, 0, 1)

def save_image(tensor, path):
    """保存图像"""
    from torchvision.transforms import ToPILImage
    img = denormalize(tensor).squeeze(0).cpu().detach()
    img = torch.clamp(img, 0, 1)
    img = ToPILImage()(img)
    img.save(path, quality=95)

def compute_gram_matrix(tensor):
    """计算Gram矩阵"""
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram / (b * c * h * w)

class VGGFeatureExtractor(nn.Module):
    """增强版VGG19特征提取器"""
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # 内容层 - 使用更深的层
        self.content_layer = '22'  # conv4_2
        
        # 风格层 - 增加更多层，捕捉更丰富的风格特征
        self.style_layers = ['0', '5', '10', '19', '28']  # conv1_1~conv5_1
        
        self.layers = nn.ModuleList()
        self.layer_names = []
        for i, layer in enumerate(vgg19):
            self.layers.append(layer)
            self.layer_names.append(str(i))
            if i == 28:
                break
    
    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if str(i) == self.content_layer:
                features['content'] = x
            if str(i) in self.style_layers:
                features[f'style_{i}'] = x
        return features

def apply_style_transfer(content_path, style_path, output_path, epochs=300, 
                         content_weight=1e5, style_weight=1e10, 
                         progress_callback=None, stop_event=None,
                         strong_mode=False):
    """增强版风格迁移 - 使用Adam优化器更稳定"""
    import torchvision.transforms as transforms
    
    # 加载图像
    content_tensor = load_image(content_path)
    style_tensor = load_image(style_path)
    
    # 强力模式参数
    if strong_mode:
        content_weight = 1e4   # 大幅降低内容保留
        style_weight = 1e12    # 大幅提高风格强度
        lr = 0.01
    else:
        lr = 0.003
    
    # 初始化 - 使用内容图（保留结构）
    generated = content_tensor.clone().requires_grad_(True)
    
    extractor = VGGFeatureExtractor().to(DEVICE)
    
    for param in extractor.parameters():
        param.requires_grad = False
    
    # 提取特征
    content_features = extractor(content_tensor)
    style_features = extractor(style_tensor)
    
    # 计算风格Gram矩阵
    style_grams = {}
    for key, value in style_features.items():
        style_grams[key] = compute_gram_matrix(value)
    
    # Adam优化器（更稳定）
    optimizer = optim.Adam([generated], lr=lr, betas=(0.9, 0.999))
    
    # 学习率调度 - 逐渐降低
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.5)
    
    # 训练循环
    for epoch in range(epochs):
        if stop_event and stop_event.is_set():
            return False
        
        gen_features = extractor(generated)
        
        # 内容损失
        content_loss = nn.MSELoss()(gen_features['content'], content_features['content'])
        
        # 风格损失 - 加权求和
        style_loss = 0
        # 深层特征权重更高，增强风格表现
        weights = {'style_0': 0.1, 'style_5': 0.15, 'style_10': 0.25, 'style_19': 0.25, 'style_28': 0.25}
        
        for key in style_grams:
            gen_gram = compute_gram_matrix(gen_features[key])
            layer_loss = nn.MSELoss()(gen_gram, style_grams[key])
            style_loss += weights.get(key, 0.2) * layer_loss
        
        # 总损失
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 每10轮报告一次
        if progress_callback and (epoch + 1) % 10 == 0:
            progress_callback(epoch + 1, epochs, total_loss.item())
    
    # 后处理 - 增强对比度和色彩
    result = denormalize(generated).squeeze(0).cpu().detach()
    result = torch.clamp(result, 0, 1)
    
    # 增强色彩饱和度
    if strong_mode:
        result = torch.pow(result, 0.9)  # 轻微提亮
        result = torch.clamp(result * 1.1, 0, 1)  # 增强对比度
    
    # 保存结果
    from torchvision.transforms import ToPILImage
    img = ToPILImage()(result)
    img.save(output_path)
    return True

# ======================== GUI类 ========================
class StyleTransferGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI艺术创作工作室 v3.1")
        self.root.geometry("1100x750")
        self.root.configure(bg="#2d2d2d")
        
        self.content_path = None
        self.style_path = None
        self.output_dir = os.path.join(PROJECT_ROOT, "output")
        self.is_training = False
        self.stop_event = threading.Event()
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        # 标题
        title_label = tk.Label(self.root, text="🎨 AI艺术创作工作室", 
                               font=("Microsoft YaHei", 24, "bold"),
                               bg="#2d2d2d", fg="#ffffff")
        title_label.pack(pady=15)
        
        # 主框架
        main_frame = tk.Frame(self.root, bg="#2d2d2d")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左侧面板 - 图片预览
        left_frame = tk.Frame(main_frame, bg="#3d3d3d", width=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # 图片预览区域
        preview_title = tk.Label(left_frame, text="图片预览", font=("Microsoft YaHei", 12),
                                 bg="#3d3d3d", fg="#ffffff")
        preview_title.pack(pady=5)
        
        # 使用 Canvas 代替 Label，支持更大尺寸
        self.preview_canvas = tk.Canvas(left_frame, width=480, height=260, bg="#4d4d4d", highlightthickness=0)
        self.preview_canvas.pack(pady=10)
        self.preview_photo = None
        
        # 结果预览
        result_title = tk.Label(left_frame, text="迁移结果", font=("Microsoft YaHei", 12),
                                bg="#3d3d3d", fg="#ffffff")
        result_title.pack(pady=5)
        
        self.result_canvas = tk.Canvas(left_frame, width=480, height=260, bg="#4d4d4d", highlightthickness=0)
        self.result_canvas.pack(pady=10)
        self.result_photo = None
        
        # 右侧面板 - 控制
        right_frame = tk.Frame(main_frame, bg="#3d3d3d", width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        control_title = tk.Label(right_frame, text="控制面板", font=("Microsoft YaHei", 12),
                                 bg="#3d3d3d", fg="#ffffff")
        control_title.pack(pady=5)
        
        # 按钮区域
        btn_frame = tk.Frame(right_frame, bg="#3d3d3d")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="📷 选择内容图片", command=self.select_content,
                  bg="#4a90d9", fg="white", font=("Microsoft YaHei", 10),
                  width=18, height=2).grid(row=0, column=0, padx=5, pady=5)
        
        tk.Button(btn_frame, text="🎨 选择风格图片", command=self.select_style,
                  bg="#4a90d9", fg="white", font=("Microsoft YaHei", 10),
                  width=18, height=2).grid(row=0, column=1, padx=5, pady=5)
        
        # 预设风格选择
        style_frame = tk.Frame(right_frame, bg="#3d3d3d")
        style_frame.pack(pady=5)
        
        tk.Label(style_frame, text="或选择预设风格:", bg="#3d3d3d", fg="#ffffff",
                 font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=5)
        
        self.style_var = tk.StringVar()
        style_combo = ttk.Combobox(style_frame, textvariable=self.style_var,
                                   values=list(PRESET_STYLES.keys()), width=12,
                                   font=("Microsoft YaHei", 10))
        style_combo.pack(side=tk.LEFT, padx=5)
        style_combo.bind("<<ComboboxSelected>>", self.on_preset_selected)
        
        # 参数设置
        param_frame = tk.LabelFrame(right_frame, text="参数设置", bg="#3d3d3d",
                                    fg="#ffffff", font=("Microsoft YaHei", 10))
        param_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # 迭代次数
        tk.Label(param_frame, text="迭代次数:", bg="#3d3d3d", fg="#ffffff").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=500)  # 默认500次，效果更好
        tk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(
            row=0, column=1, padx=5, pady=5)
        
        # 风格强度
        tk.Label(param_frame, text="风格强度:", bg="#3d3d3d", fg="#ffffff").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.style_intensity = tk.DoubleVar(value=1.0)
        style_scale = tk.Scale(param_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
                               variable=self.style_intensity, bg="#3d3d3d", fg="white",
                               length=120, showvalue=1, digits=1)
        style_scale.grid(row=1, column=1, padx=5, pady=5)
        
        # 强力模式
        self.strong_mode = tk.BooleanVar(value=False)
        strong_btn = tk.Checkbutton(param_frame, text="强力迁移模式 💥", variable=self.strong_mode,
                       bg="#3d3d3d", fg="#ff6600", activebackground="#3d3d3d", activeforeground="#ff6600",
                       selectcolor="#4d4d4d", font=("Microsoft YaHei", 10, "bold"))
        strong_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky=tk.W, padx=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var,
                                            maximum=100, length=300)
        self.progress_bar.pack(pady=10)
        
        # 状态标签
        self.status_label = tk.Label(right_frame, text="就绪", bg="#3d3d3d",
                                      fg="#00ff00", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        # 开始/停止按钮
        self.start_btn = tk.Button(right_frame, text="🚀 开始风格迁移", 
                                    command=self.start_transfer,
                                    bg="#28a745", fg="white",
                                    font=("Microsoft YaHei", 14, "bold"),
                                    width=20, height=2)
        self.start_btn.pack(pady=15)
        
        # 保存按钮
        tk.Button(right_frame, text="💾 保存结果", command=self.save_result,
                  bg="#ffc107", fg="black", font=("Microsoft YaHei", 10),
                  width=15, height=1).pack(pady=5)
    
    def select_content(self):
        """选择内容图片"""
        path = filedialog.askopenfilename(
            title="选择内容图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.content_path = path
            self.update_preview()
            self.status_label.config(text=f"已选择内容图: {os.path.basename(path)}", fg="#00ff00")
    
    def select_style(self):
        """选择风格图片"""
        path = filedialog.askopenfilename(
            title="选择风格图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.style_path = path
            self.style_var.set("")  # 清除预设选择
            self.update_preview()
            self.status_label.config(text=f"已选择风格图: {os.path.basename(path)}", fg="#00ff00")
    
    def on_preset_selected(self, event):
        """预设风格选中"""
        style_name = self.style_var.get()
        if style_name and style_name in PRESET_STYLES:
            style_info = PRESET_STYLES[style_name]
            style_dir = os.path.join(PROJECT_ROOT, style_info["folder"])
            # 随机选择一个风格图
            import random
            style_file = random.choice(style_info["files"])
            self.style_path = os.path.join(style_dir, style_file)
            self.update_preview()
            self.status_label.config(text=f"已选择预设风格: {style_name} ({style_file})", fg="#00ff00")
    
    def update_preview(self):
        """更新预览 - 使用Canvas显示"""
        if self.content_path and self.style_path:
            try:
                # 加载图片
                content_img = Image.open(self.content_path)
                style_img = Image.open(self.style_path)
                
                # 固定预览尺寸：200x200，正方形裁切居中
                preview_size = 200
                
                def center_crop_square(img, size):
                    """居中裁切为正方形"""
                    w, h = img.size
                    min_dim = min(w, h)
                    left = (w - min_dim) // 2
                    top = (h - min_dim) // 2
                    img = img.crop((left, top, left + min_dim, top + min_dim))
                    return img.resize((size, size), Image.Resampling.LANCZOS)
                
                content_img = center_crop_square(content_img, preview_size)
                style_img = center_crop_square(style_img, preview_size)
                
                # 转换为 PhotoImage
                self.preview_photo = ImageTk.PhotoImage(content_img)
                self.preview_canvas.create_image(10, 10, anchor=tk.NW, image=self.preview_photo)
                
                self.preview_photo2 = ImageTk.PhotoImage(style_img)
                self.preview_canvas.create_image(250, 10, anchor=tk.NW, image=self.preview_photo2)
                
                # 添加文字标签
                self.preview_canvas.create_text(10 + preview_size//2, preview_size + 15, 
                                                text="内容图", fill="white", anchor=tk.CENTER)
                self.preview_canvas.create_text(250 + preview_size//2, preview_size + 15, 
                                                text="风格图", fill="white", anchor=tk.CENTER)
            except Exception as e:
                self.status_label.config(text=f"预览失败: {e}", fg="#ff0000")
    
    def start_transfer(self):
        """开始风格迁移"""
        if self.is_training:
            # 停止
            self.stop_event.set()
            self.status_label.config(text="正在停止...", fg="#ff0000")
            return
        
        if not self.content_path:
            messagebox.showwarning("警告", "请先选择内容图片！")
            return
        if not self.style_path:
            messagebox.showwarning("警告", "请先选择风格图片或预设风格！")
            return
        
        self.is_training = True
        self.stop_event.clear()
        self.start_btn.config(text="⏹ 停止", bg="#dc3545")
        self.status_label.config(text="正在训练...", fg="#ffff00")
        
        # 获取参数
        epochs = self.epochs_var.get()
        intensity = self.style_intensity.get()
        
        if self.strong_mode.get():
            content_weight = 5e4 * (1 / intensity)
            style_weight = 5e11 * intensity
        else:
            content_weight = 1e5 * (1 / intensity)
            style_weight = 1e10 * intensity
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"result_{timestamp}.jpg")
        
        def train():
            try:
                def progress_callback(epoch, total, loss):
                    progress = (epoch / total) * 100
                    self.root.after(0, lambda: self.progress_var.set(progress))
                    loss_str = f"{loss:.2e}" if loss > 0 else "计算中..."
                    self.root.after(0, lambda: self.status_label.config(
                        text=f"进度: {epoch}/{total} | 损失: {loss_str}", fg="#ffff00"))
                
                success = apply_style_transfer(
                    self.content_path, self.style_path, output_path,
                    epochs=epochs, content_weight=content_weight, style_weight=style_weight,
                    progress_callback=progress_callback, stop_event=self.stop_event,
                    strong_mode=self.strong_mode.get()
                )
                
                self.root.after(0, lambda: self.on_training_complete(output_path, success))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"错误: {e}", fg="#ff0000"))
                self.root.after(0, self.reset_ui)
        
        thread = threading.Thread(target=train, daemon=True)
        thread.start()
    
    def on_training_complete(self, output_path, success):
        """训练完成"""
        self.last_output = output_path
        if success:
            self.status_label.config(text="✅ 训练完成!", fg="#00ff00")
            try:
                result_img = Image.open(output_path)
                # 自适应缩放结果图
                target_size = 220
                if result_img.width <= target_size and result_img.height <= target_size:
                    new_size = result_img.size
                elif result_img.width > result_img.height:
                    new_size = (target_size, int(target_size * result_img.height / result_img.width))
                else:
                    new_size = (int(target_size * result_img.width / result_img.height), target_size)
                
                result_img = result_img.resize(new_size, Image.Resampling.LANCZOS)
                
                self.result_photo = ImageTk.PhotoImage(result_img)
                self.result_canvas.create_image(10, 10, anchor=tk.NW, image=self.result_photo)
                self.result_canvas.create_text(10 + new_size[0]//2, new_size[1] + 15, 
                                              text="迁移结果", fill="#00ff00", anchor=tk.CENTER)
                
                messagebox.showinfo("完成", f"风格迁移完成！\n结果已保存到:\n{output_path}")
            except Exception as e:
                messagebox.showinfo("完成", f"风格迁移完成！\n结果已保存到:\n{output_path}")
        else:
            self.status_label.config(text="已停止", fg="#ff0000")
        self.reset_ui()
    
    def reset_ui(self):
        """重置UI"""
        self.is_training = False
        self.start_btn.config(text="🚀 开始风格迁移", bg="#28a745")
        self.progress_var.set(0)
    
    def save_result(self):
        """保存结果"""
        if hasattr(self, 'last_output') and self.last_output:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
            )
            if save_path:
                Image.open(self.last_output).save(save_path)
                messagebox.showinfo("成功", f"已保存到: {save_path}")
        else:
            messagebox.showwarning("警告", "没有可保存的结果！")

# ======================== 主入口 ========================
def main():
    """主函数"""
    root = tk.Tk()
    app = StyleTransferGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
