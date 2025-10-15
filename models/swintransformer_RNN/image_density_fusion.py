"""
Image-Density Fusion Module
图像密度融合模块，用于融合原始图像和密度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    """
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class LocalFeatureExtraction(nn.Module):
    """
    局部特征提取模块
    """
    def __init__(self, in_channels, out_channels):
        super(LocalFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 通道注意力
        self.channel_attention = ChannelAttention(out_channels)
        # 空间注意力
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        # 局部特征提取
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 应用通道注意力
        out = out * self.channel_attention(out)
        
        # 应用空间注意力
        out = out * self.spatial_attention(out)
        
        out = self.relu(out)
        return out


class MultiScaleFeatureAggregation(nn.Module):
    """
    多尺度特征聚合模块
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureAggregation, self).__init__()
        
        # 不同尺度的卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels//4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels//4, 7, padding=3)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 多尺度特征提取
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3(x)
        feat3 = self.conv5x5(x)
        feat4 = self.conv7x7(x)
        
        # 特征拼接
        multi_scale_feat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        multi_scale_feat = self.bn(multi_scale_feat)
        multi_scale_feat = self.relu(multi_scale_feat)
        
        # 特征融合
        fused_feat = self.fusion_conv(multi_scale_feat)
        fused_feat = self.fusion_bn(fused_feat)
        fused_feat = self.fusion_relu(fused_feat)
        
        return fused_feat


class ImageDensityFusion(nn.Module):
    """
    图像密度融合模块
    融合原始图像和密度特征，包含：
    1. 图像拼接（原始图像 + 密度单一通道）
    2. 局部特征提取 + 通道注意力
    3. 多尺度特征聚合
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 density_channels: int = 1,
                 feature_channels: int = 64,
                 output_channels: int = 512):
        """
        初始化图像密度融合模块
        
        Args:
            input_channels: 输入图像通道数
            density_channels: 密度特征通道数
            feature_channels: 特征通道数
            output_channels: 输出通道数
        """
        super(ImageDensityFusion, self).__init__()
        
        self.input_channels = input_channels
        self.density_channels = density_channels
        self.feature_channels = feature_channels
        self.output_channels = output_channels
        
        # 输入通道数 = 原始图像通道数 + 密度通道数
        total_input_channels = input_channels + density_channels
        
        # 1. 初始特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(total_input_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. 局部特征提取 + 通道注意力
        self.local_feature_extraction = LocalFeatureExtraction(
            feature_channels, feature_channels
        )
        
        # 3. 多尺度特征聚合
        self.multi_scale_aggregation = MultiScaleFeatureAggregation(
            feature_channels, feature_channels * 2
        )
        
        # 4. 最终特征映射
        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_channels * 2, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        )
        
        # 输出特征维度
        self.output_dim = output_channels
    
    def forward(self, original_image: torch.Tensor, density_channel: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            original_image: 原始图像 [B, C, H, W]
            density_channel: 密度单一通道 [B, 1, H, W]
            
        Returns:
            fused_features: 融合后的特征 [B, output_channels]
        """
        # 1. 图像拼接：原始图像 + 密度单一通道
        if original_image.shape[-2:] != density_channel.shape[-2:]:
            # 调整密度通道尺寸以匹配原始图像
            density_channel = F.interpolate(
                density_channel, 
                size=original_image.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 拼接原始图像和密度通道
        concatenated = torch.cat([original_image, density_channel], dim=1)
        
        # 2. 初始特征提取
        features = self.initial_conv(concatenated)
        
        # 3. 局部特征提取 + 通道注意力
        local_features = self.local_feature_extraction(features)
        
        # 4. 多尺度特征聚合
        multi_scale_features = self.multi_scale_aggregation(local_features)
        
        # 5. 最终特征映射
        fused_features = self.final_conv(multi_scale_features)
        
        # 展平
        fused_features = fused_features.view(fused_features.size(0), -1)
        
        return fused_features
    
    def forward_batch(self, video_frames: torch.Tensor, density_channels: torch.Tensor) -> torch.Tensor:
        """
        批量前向传播
        
        Args:
            video_frames: 视频帧 [B, T, C, H, W]
            density_channels: 密度通道 [B, T, 1, H, W]
            
        Returns:
            fused_features: 融合后的特征 [B, T, output_channels]
        """
        batch_size, num_frames = video_frames.shape[:2]
        device = video_frames.device
        
        # 重塑为批量处理格式
        frames_flat = video_frames.view(-1, *video_frames.shape[2:])  # [B*T, C, H, W]
        density_flat = density_channels.view(-1, *density_channels.shape[2:])  # [B*T, 1, H, W]
        
        # 批量处理
        fused_features_flat = self.forward(frames_flat, density_flat)  # [B*T, output_channels]
        
        # 重塑回时间序列格式
        fused_features = fused_features_flat.view(batch_size, num_frames, -1)  # [B, T, output_channels]
        
        return fused_features


def create_image_density_fusion(input_channels: int = 3,
                              density_channels: int = 1,
                              feature_channels: int = 64,
                              output_channels: int = 512) -> ImageDensityFusion:
    """
    创建图像密度融合模块
    
    Args:
        input_channels: 输入图像通道数
        density_channels: 密度特征通道数
        feature_channels: 特征通道数
        output_channels: 输出通道数
        
    Returns:
        fusion_module: 图像密度融合模块
    """
    return ImageDensityFusion(
        input_channels=input_channels,
        density_channels=density_channels,
        feature_channels=feature_channels,
        output_channels=output_channels
    )


# 使用示例
if __name__ == "__main__":
    # 创建图像密度融合模块
    fusion_module = create_image_density_fusion()
    
    # 测试数据
    batch_size = 2
    original_image = torch.randn(batch_size, 3, 224, 224)
    density_channel = torch.randn(batch_size, 1, 224, 224)
    
    # 前向传播
    with torch.no_grad():
        fused_features = fusion_module(original_image, density_channel)
        print(f"原始图像形状: {original_image.shape}")
        print(f"密度通道形状: {density_channel.shape}")
        print(f"融合特征形状: {fused_features.shape}")
        print(f"输出特征维度: {fusion_module.output_dim}")
