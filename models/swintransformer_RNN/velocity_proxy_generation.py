"""
Velocity Proxy Generation Module for Video Classification
该模块用于生成视频的速度代理特征，通过光流估计和幅度计算来捕获运动信息
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


class VelocityProxyGenerator:
    """
    Velocity Proxy Generation Module
    使用光流估计和幅度计算生成视频的速度代理特征
    包含：光流估计、幅度计算、归一化、速度代理图生成
    """
    
    def __init__(self, 
                 input_size: int = 224,
                 velocity_map_size: int = 32,
                 flow_method: str = 'farneback',
                 flow_params: dict = None):
        """
        初始化速度代理生成器
        
        Args:
            input_size: 输入图像尺寸
            velocity_map_size: 速度图尺寸
            flow_method: 光流估计方法 ('farneback', 'lucas_kanade', 'deepflow')
            flow_params: 光流参数
        """
        self.input_size = input_size
        self.velocity_map_size = velocity_map_size
        self.flow_method = flow_method
        
        # 默认光流参数
        if flow_params is None:
            if flow_method == 'farneback':
                self.flow_params = {
                    'pyr_scale': 0.5,
                    'levels': 3,
                    'winsize': 15,
                    'iterations': 3,
                    'poly_n': 5,
                    'poly_sigma': 1.2,
                    'flags': 0
                }
            elif flow_method == 'lucas_kanade':
                self.flow_params = {
                    'max_corners': 100,
                    'quality_level': 0.3,
                    'min_distance': 7,
                    'block_size': 7
                }
            else:
                self.flow_params = {}
        else:
            self.flow_params = flow_params
        
        # 输出特征维度
        self.output_dim = velocity_map_size * velocity_map_size
    
    def grayscale_conversion(self, image: np.ndarray) -> np.ndarray:
        """
        灰度转换
        
        Args:
            image: 输入图像 [H, W, C]
            
        Returns:
            gray_image: 灰度图像 [H, W]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return gray
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        计算两帧之间的光流
        
        Args:
            frame1: 前一帧 [H, W]
            frame2: 后一帧 [H, W]
            
        Returns:
            optical_flow: 光流 [H, W, 2]
        """
        if self.flow_method == 'farneback':
            # Farneback光流方法
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None,
                pyr_scale=self.flow_params['pyr_scale'],
                levels=self.flow_params['levels'],
                winsize=self.flow_params['winsize'],
                iterations=self.flow_params['iterations'],
                poly_n=self.flow_params['poly_n'],
                poly_sigma=self.flow_params['poly_sigma'],
                flags=self.flow_params['flags']
            )
        elif self.flow_method == 'lucas_kanade':
            # Lucas-Kanade光流方法
            # 检测角点
            corners = cv2.goodFeaturesToTrack(
                frame1,
                maxCorners=self.flow_params['max_corners'],
                qualityLevel=self.flow_params['quality_level'],
                minDistance=self.flow_params['min_distance'],
                blockSize=self.flow_params['block_size']
            )
            
            if corners is not None and len(corners) > 0:
                # 计算光流
                next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                    frame1, frame2, corners, None
                )
                
                # 选择有效的点
                good_corners = corners[status == 1]
                good_next = next_corners[status == 1]
                
                if len(good_corners) > 0:
                    # 计算位移向量
                    displacement = good_next - good_corners
                    
                    # 创建光流图
                    flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
                    
                    # 将位移向量映射到光流图
                    for i, (corner, disp) in enumerate(zip(good_corners, displacement)):
                        x, y = int(corner[0, 0]), int(corner[0, 1])
                        if 0 <= x < frame1.shape[1] and 0 <= y < frame1.shape[0]:
                            flow[y, x, 0] = disp[0, 0]  # u分量
                            flow[y, x, 1] = disp[0, 1]  # v分量
                else:
                    flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
            else:
                flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
        else:
            # 默认使用Farneback方法
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        return flow
    
    def compute_magnitude(self, optical_flow: np.ndarray) -> np.ndarray:
        """
        计算光流的幅度
        
        Args:
            optical_flow: 光流 [H, W, 2]
            
        Returns:
            magnitude: 幅度图 [H, W]
        """
        # 分离u和v分量
        u = optical_flow[:, :, 0]
        v = optical_flow[:, :, 1]
        
        # 计算幅度
        magnitude = np.sqrt(u**2 + v**2)
        
        return magnitude
    
    def normalize_velocity(self, magnitude: np.ndarray) -> np.ndarray:
        """
        归一化速度幅度
        
        Args:
            magnitude: 幅度图 [H, W]
            
        Returns:
            normalized_velocity: 归一化后的速度图 [H, W]
        """
        # 避免除零
        max_magnitude = np.max(magnitude)
        if max_magnitude > 0:
            normalized = magnitude / max_magnitude
        else:
            normalized = magnitude
        
        # 确保值在[0, 1]范围内
        normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def compute_velocity_proxy_map(self, velocity_map: np.ndarray) -> np.ndarray:
        """
        计算速度代理图
        
        Args:
            velocity_map: 速度图 [H, W]
            
        Returns:
            velocity_proxy: 速度代理图 [velocity_map_size, velocity_map_size]
        """
        # 调整到目标尺寸
        resized = cv2.resize(velocity_map, (self.velocity_map_size, self.velocity_map_size))
        
        # 归一化到0-1范围
        velocity_proxy = resized.astype(np.float32)
        
        return velocity_proxy
    
    def process_frame_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        处理帧对，生成速度代理特征
        
        Args:
            frame1: 前一帧 [H, W, C] 或 [H, W]
            frame2: 后一帧 [H, W, C] 或 [H, W]
            
        Returns:
            velocity_proxy: 速度代理特征 [velocity_map_size, velocity_map_size]
        """
        # 1. 灰度转换
        gray1 = self.grayscale_conversion(frame1)
        gray2 = self.grayscale_conversion(frame2)
        
        # 2. 光流估计
        optical_flow = self.compute_optical_flow(gray1, gray2)
        
        # 3. 幅度计算
        magnitude = self.compute_magnitude(optical_flow)
        
        # 4. 归一化
        normalized_velocity = self.normalize_velocity(magnitude)
        
        # 5. 生成速度代理图
        velocity_proxy = self.compute_velocity_proxy_map(normalized_velocity)
        
        return velocity_proxy
    
    def process_batch_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        批量处理视频帧，生成速度代理特征
        
        Args:
            frames: 视频帧 [B, T, C, H, W]
            
        Returns:
            velocity_channels: 速度通道 [B, T-1, 1, H, W]
        """
        batch_size, num_frames, channels, height, width = frames.shape
        device = frames.device
        
        # 转换为numpy进行处理
        frames_np = frames.cpu().numpy().transpose(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        
        all_velocity_channels = []
        
        # 处理连续帧对
        for t in range(num_frames - 1):
            frame1_batch = frames_np[:, t]  # [B, H, W, C]
            frame2_batch = frames_np[:, t + 1]  # [B, H, W, C]
            
            # 批量处理当前时间步的所有样本
            batch_velocity_channels = []
            for b in range(batch_size):
                frame1 = frame1_batch[b]  # [H, W, C]
                frame2 = frame2_batch[b]  # [H, W, C]
                velocity_channel = self.process_frame_pair(frame1, frame2)  # [velocity_map_size, velocity_map_size]
                batch_velocity_channels.append(velocity_channel)
            
            # 转换为tensor并调整尺寸
            velocity_channel = torch.from_numpy(np.array(batch_velocity_channels)).float().to(device)  # [B, velocity_map_size, velocity_map_size]
            
            # 调整到原始图像尺寸
            velocity_channel = F.interpolate(
                velocity_channel.unsqueeze(1), 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)  # [B, H, W]
            
            velocity_channel = velocity_channel.unsqueeze(1)  # [B, 1, H, W]
            all_velocity_channels.append(velocity_channel)
        
        # 堆叠时间维度
        velocity_channels = torch.stack(all_velocity_channels, dim=1)  # [B, T-1, 1, H, W]
        
        return velocity_channels
    
    def process_video_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        处理视频帧序列，生成速度代理特征
        
        Args:
            frames: 视频帧 [B, T, C, H, W]
            
        Returns:
            velocity_proxies: 速度代理特征 [B, output_dim]
        """
        batch_size, num_frames = frames.shape[:2]
        device = frames.device
        
        # 转换为numpy进行处理
        frames_np = frames.cpu().numpy().transpose(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        
        all_velocity_proxies = []
        
        for b in range(batch_size):
            frame_velocities = []
            for t in range(num_frames - 1):
                frame1 = frames_np[b, t]
                frame2 = frames_np[b, t + 1]
                velocity_proxy = self.process_frame_pair(frame1, frame2)
                frame_velocities.append(velocity_proxy)
            
            # 对时间维度进行平均
            avg_velocity = np.mean(frame_velocities, axis=0)
            all_velocity_proxies.append(avg_velocity)
        
        # 转换为tensor
        velocity_proxies = torch.from_numpy(np.array(all_velocity_proxies)).float().to(device)
        
        # 展平
        velocity_proxies = velocity_proxies.view(batch_size, -1)
        
        return velocity_proxies
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        调用处理函数
        
        Args:
            frames: 视频帧 [B, T, C, H, W]
            
        Returns:
            velocity_proxy: 速度代理特征 [B, output_dim]
        """
        return self.process_video_frames(frames)


def create_velocity_generator(config: dict = None) -> VelocityProxyGenerator:
    """
    创建速度代理生成器
    
    Args:
        config: 配置字典
        
    Returns:
        velocity_generator: 速度代理生成器
    """
    if config is None:
        config = {}
    
    return VelocityProxyGenerator(
        input_size=config.get('input_size', 224),
        velocity_map_size=config.get('velocity_map_size', 32),
        flow_method=config.get('flow_method', 'farneback'),
        flow_params=config.get('flow_params', None)
    )


# 使用示例
if __name__ == "__main__":
    # 创建速度代理生成器
    generator = create_velocity_generator()
    
    # 测试
    batch_size = 2
    num_frames = 16
    test_frames = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    with torch.no_grad():
        velocity_features = generator.process_batch_frames(test_frames)
        print(f"Input shape: {test_frames.shape}")
        print(f"Velocity channels shape: {velocity_features.shape}")
        print(f"Feature dimension: {generator.output_dim}")
