"""
Density Proxy Generation Module for Video Classification
该模块用于生成视频的密度代理特征，用于增强视频分类性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional
import os
import glob
from tqdm import tqdm


class DensityProxyGenerator:
    """
    Density Proxy Generation Module
    使用传统计算机视觉方法生成视频的密度代理特征
    包含：灰度转换、高斯模糊、局部对比度增强、密度代理图生成
    """
    
    def __init__(self, 
                 input_size: int = 224,
                 density_map_size: int = 32,
                 gaussian_kernel_size: int = 5,
                 gaussian_sigma: float = 1.0,
                 contrast_alpha: float = 1.5,
                 contrast_beta: int = 30):
        """
        初始化密度代理生成器
        
        Args:
            input_size: 输入图像尺寸
            density_map_size: 密度图尺寸
            gaussian_kernel_size: 高斯核大小
            gaussian_sigma: 高斯标准差
            contrast_alpha: 对比度增强参数
            contrast_beta: 对比度增强偏移
        """
        self.input_size = input_size
        self.density_map_size = density_map_size
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.contrast_alpha = contrast_alpha
        self.contrast_beta = contrast_beta
        
        # 输出特征维度
        self.output_dim = density_map_size * density_map_size
    
    def grayscale_conversion(self, image: np.ndarray) -> np.ndarray:
        """
        灰度转换
        
        Args:
            image: 输入图像 [H, W, C]
            
        Returns:
            gray_image: 灰度图像 [H, W]
        """
        if len(image.shape) == 3:
            # 使用加权平均进行灰度转换
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return gray
    
    def gaussian_blur_filter(self, image: np.ndarray) -> np.ndarray:
        """
        高斯模糊滤波
        
        Args:
            image: 输入图像 [H, W]
            
        Returns:
            blurred_image: 模糊后的图像 [H, W]
        """
        return cv2.GaussianBlur(
            image, 
            (self.gaussian_kernel_size, self.gaussian_kernel_size), 
            self.gaussian_sigma
        )
    
    def local_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        局部对比度增强
        
        Args:
            image: 输入图像 [H, W]
            
        Returns:
            enhanced_image: 增强后的图像 [H, W]
        """
        # 使用CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # 额外的对比度调整
        enhanced = cv2.convertScaleAbs(enhanced, alpha=self.contrast_alpha, beta=self.contrast_beta)
        
        return enhanced
    
    def compute_density_proxy_map(self, image: np.ndarray) -> np.ndarray:
        """
        计算密度代理图
        
        Args:
            image: 输入图像 [H, W]
            
        Returns:
            density_map: 密度代理图 [density_map_size, density_map_size]
        """
        # 调整到目标尺寸
        resized = cv2.resize(image, (self.density_map_size, self.density_map_size))
        
        # 归一化到0-1范围
        density_map = resized.astype(np.float32) / 255.0
        
        return density_map
    
    def process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧图像，生成密度代理特征
        
        Args:
            frame: 输入帧 [H, W, C] 或 [H, W]
            
        Returns:
            density_proxy: 密度代理特征 [density_map_size, density_map_size]
        """
        # 1. 灰度转换
        gray = self.grayscale_conversion(frame)
        
        # 2. 高斯模糊滤波
        blurred = self.gaussian_blur_filter(gray)
        
        # 3. 局部对比度增强
        enhanced = self.local_contrast_enhancement(blurred)
        
        # 4. 生成密度代理图
        density_proxy = self.compute_density_proxy_map(enhanced)
        
        return density_proxy
    
    def process_batch_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        批量处理视频帧，生成密度代理特征
        
        Args:
            frames: 视频帧 [B, T, C, H, W]
            
        Returns:
            density_channels: 密度通道 [B, T, 1, H, W]
        """
        batch_size, num_frames, channels, height, width = frames.shape
        device = frames.device
        
        # 转换为numpy进行处理
        frames_np = frames.cpu().numpy().transpose(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        
        all_density_channels = []
        
        for t in range(num_frames):
            frame_batch = frames_np[:, t]  # [B, H, W, C]
            
            # 批量处理当前时间步的所有样本
            batch_density_channels = []
            for b in range(batch_size):
                single_frame = frame_batch[b]  # [H, W, C]
                density_channel = self.process_single_frame(single_frame)  # [density_map_size, density_map_size]
                batch_density_channels.append(density_channel)
            
            # 转换为tensor并调整尺寸
            density_channel = torch.from_numpy(np.array(batch_density_channels)).float().to(device)  # [B, density_map_size, density_map_size]
            
            # 调整到原始图像尺寸
            density_channel = F.interpolate(
                density_channel.unsqueeze(1), 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)  # [B, H, W]
            
            density_channel = density_channel.unsqueeze(1)  # [B, 1, H, W]
            all_density_channels.append(density_channel)
        
        # 堆叠时间维度
        density_channels = torch.stack(all_density_channels, dim=1)  # [B, T, 1, H, W]
        
        return density_channels
    
    def process_video_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        处理视频帧序列，生成密度代理特征
        
        Args:
            frames: 视频帧 [B, T, C, H, W]
            
        Returns:
            density_proxies: 密度代理特征 [B, output_dim]
        """
        batch_size, num_frames = frames.shape[:2]
        device = frames.device
        
        # 转换为numpy进行处理
        frames_np = frames.cpu().numpy().transpose(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        
        all_density_proxies = []
        
        for b in range(batch_size):
            frame_densities = []
            for t in range(num_frames):
                frame = frames_np[b, t]
                density_proxy = self.process_single_frame(frame)
                frame_densities.append(density_proxy)
            
            # 对时间维度进行平均
            avg_density = np.mean(frame_densities, axis=0)
            all_density_proxies.append(avg_density)
        
        # 转换为tensor
        density_proxies = torch.from_numpy(np.array(all_density_proxies)).float().to(device)
        
        # 展平
        density_proxies = density_proxies.view(batch_size, -1)
        
        return density_proxies
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        调用处理函数
        
        Args:
            frames: 视频帧 [B, T, C, H, W]
            
        Returns:
            density_proxy: 密度代理特征 [B, output_dim]
        """
        return self.process_video_frames(frames)


def density_extract(video_folders: List[str], 
                   data_path: str,
                   selected_frames: List[int],
                   transform=None,
                   density_generator: Optional[DensityProxyGenerator] = None,
                   device: str = 'cuda',
                   batch_size: int = 32) -> np.ndarray:
    """
    批量提取视频的密度代理特征
    
    Args:
        video_folders: 视频文件夹路径列表
        data_path: 数据根路径
        selected_frames: 选择的帧索引
        transform: 图像变换
        density_generator: 密度代理生成器
        device: 设备
        batch_size: 批处理大小
        
    Returns:
        density_features: 密度代理特征 [N, feature_dim]
    """
    if density_generator is None:
        density_generator = DensityProxyGenerator()
    
    density_generator = density_generator.to(device)
    density_generator.eval()
    
    all_features = []
    
    # 批量处理
    for i in tqdm(range(0, len(video_folders), batch_size), desc="Extracting density features"):
        batch_folders = video_folders[i:i+batch_size]
        batch_frames = []
        
        # 加载批次数据
        for folder in batch_folders:
            frames = []
            for frame_idx in selected_frames:
                frame_path = os.path.join(data_path, folder, f"frame{frame_idx:06d}.jpg")
                if os.path.exists(frame_path):
                    # 加载图像
                    import torchvision.transforms as transforms
                    from PIL import Image
                    
                    image = Image.open(frame_path).convert('RGB')
                    if transform:
                        image = transform(image)
                    else:
                        # 默认变换
                        default_transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
                        ])
                        image = default_transform(image)
                    
                    frames.append(image)
                else:
                    # 如果帧不存在，使用零填充
                    frames.append(torch.zeros(3, 224, 224))
            
            if frames:
                batch_frames.append(torch.stack(frames))
            else:
                # 如果所有帧都不存在，使用零填充
                batch_frames.append(torch.zeros(len(selected_frames), 3, 224, 224))
        
        if batch_frames:
            # 转换为tensor并移动到设备
            batch_tensor = torch.stack(batch_frames).to(device)
            
            # 提取密度特征
            with torch.no_grad():
                features = density_generator(batch_tensor)
                all_features.append(features.cpu().numpy())
    
    if all_features:
        return np.concatenate(all_features, axis=0)
    else:
        return np.array([])


def create_density_generator(config: dict = None) -> DensityProxyGenerator:
    """
    创建密度代理生成器
    
    Args:
        config: 配置字典
        
    Returns:
        density_generator: 密度代理生成器
    """
    if config is None:
        config = {}
    
    return DensityProxyGenerator(
        input_size=config.get('input_size', 224),
        density_map_size=config.get('density_map_size', 32),
        gaussian_kernel_size=config.get('gaussian_kernel_size', 5),
        gaussian_sigma=config.get('gaussian_sigma', 1.0),
        contrast_alpha=config.get('contrast_alpha', 1.5),
        contrast_beta=config.get('contrast_beta', 30)
    )


# 使用示例
if __name__ == "__main__":
    # 创建密度代理生成器
    generator = create_density_generator()
    
    # 测试
    batch_size = 2
    num_frames = 16
    test_frames = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    with torch.no_grad():
        features = generator(test_frames)
        print(f"Input shape: {test_frames.shape}")
        print(f"Output shape: {features.shape}")
        print(f"Feature dimension: {generator.output_dim}")
