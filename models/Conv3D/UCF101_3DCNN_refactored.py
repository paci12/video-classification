#!/usr/bin/env python3
"""
Refactored Conv3D Model for UCF101 Video Classification
使用公共组件的重构版本
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle

# 添加utils路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# 导入公共组件
from common.label_utils import labels2cat, labels2onehot, onehot2labels, cat2labels
from common.data_loaders import Dataset_3DCNN
from common.training_utils import (
    train_epoch, validate_epoch, save_checkpoint, load_checkpoint,
    get_optimizer, get_scheduler, save_predictions, count_parameters
)
from config_loader import ConfigLoader

# 配置参数
class Config:
    # 数据路径
    data_path = None
    action_name_path = None
    save_model_path = "./Conv3D_ckpt/"
    
    # 3D CNN参数
    num_classes = 101
    dropout_p = 0.5
    
    # 训练参数
    epochs = 120
    batch_size = 16  # 3D CNN需要较小的batch size
    learning_rate = 1e-4
    log_interval = 10
    
    # 帧选择 - 从配置文件读取
    begin_frame = 1
    end_frame = 17  # 默认16帧，end_frame = num_frames + 1
    skip_frame = 1

class Conv3DModel(nn.Module):
    """3D CNN模型 - 使用公共组件"""
    
    def __init__(self, config):
        super(Conv3DModel, self).__init__()
        
        # 3D CNN架构
        self.conv3d = nn.Sequential(
            # 第一个3D卷积块
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # 第二个3D卷积块
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # 第三个3D卷积块
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # 第四个3D卷积块
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            # 全局平均池化
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(config.dropout_p),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_p),
            nn.Linear(512, config.num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time, height, width)
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

def load_data(config):
    """加载数据"""
    print("Loading data...")
    
    # 加载动作名称
    with open(config.action_name_path, 'rb') as f:
        action_names = pickle.load(f)
    
    # 标签编码器
    le = LabelEncoder()
    le.fit(action_names)
    
    # One-hot编码器
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)
    
    # 获取所有视频文件（遍历类别子目录），用目录名作为标签
    actions = []
    all_names = []
    
    try:
        action_dirs = [d for d in os.listdir(config.data_path) if os.path.isdir(os.path.join(config.data_path, d))]
    except FileNotFoundError:
        raise FileNotFoundError(f"Data path not found: {config.data_path}")

    for action_dir in action_dirs:
        action_dir_path = os.path.join(config.data_path, action_dir)
        try:
            video_files = os.listdir(action_dir_path)
        except Exception:
            continue
        for f in video_files:
            if not f.startswith('v_'):
                continue
            actions.append(action_dir)
            all_names.append(os.path.join(action_dir, f))
    
    # 使用公共函数转换标签
    all_X_list = all_names
    all_y_list = labels2cat(le, actions)
    
    # 训练测试分割
    train_list, test_list, train_label, test_label = train_test_split(
        all_X_list, all_y_list, test_size=0.25, random_state=42
    )
    
    # 数据变换 - 3D CNN使用灰度图像
    transform = transforms.Compose([
        transforms.Resize([112, 112]),  # 3D CNN通常使用较小的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图像归一化
    ])
    
    # 选择帧
    selected_frames = np.arange(config.begin_frame, config.end_frame, config.skip_frame).tolist()
    
    # 使用公共数据加载器
    train_set = Dataset_3DCNN(
        data_path=config.data_path,
        folders=train_list,
        labels=train_label,
        frames=selected_frames,
        transform=transform
    )
    
    valid_set = Dataset_3DCNN(
        data_path=config.data_path,
        folders=test_list,
        labels=test_label,
        frames=selected_frames,
        transform=transform
    )
    
    # 数据加载器
    use_cuda = torch.cuda.is_available()
    params = {
        'batch_size': config.batch_size,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True
    } if use_cuda else {
        'batch_size': config.batch_size,
        'shuffle': True
    }
    
    train_loader = DataLoader(train_set, **params)
    valid_loader = DataLoader(valid_set, **params)
    
    print(f"Train samples: {len(train_set)}")
    print(f"Validation samples: {len(valid_set)}")
    
    return train_loader, valid_loader, le, enc

def train_model(config):
    """训练模型"""
    print("Starting Conv3D training...")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, valid_loader, le, enc = load_data(config)
    
    # 创建模型
    model = Conv3DModel(config).to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 使用公共函数创建优化器和调度器
    optimizer = get_optimizer(
        model, 
        optimizer_name='adam', 
        lr=config.learning_rate
    )
    
    scheduler = get_scheduler(
        optimizer, 
        scheduler_name='step', 
        step_size=20, 
        gamma=0.1
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # 使用公共训练函数
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # 使用公共验证函数
        val_loss, val_acc = validate_epoch(
            model, valid_loader, criterion, device, epoch+1
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch+1, val_loss,
                os.path.join(config.save_model_path, 'best_model.pth')
            )
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch+1, val_loss,
                os.path.join(config.save_model_path, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        scheduler.step()
    
    print("Training completed!")
    
    # 保存预测结果
    save_predictions(
        model, valid_loader, device,
        os.path.join(config.save_model_path, 'predictions.pkl')
    )
    
    return model

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Conv3D Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Override epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')
    parser.add_argument('--data_path', type=str, default=None, help='Override data path')
    
    args = parser.parse_args()
    
    # 使用ConfigLoader加载配置
    if args.config and os.path.exists(args.config):
        config_loader = ConfigLoader(args.config)
    else:
        # 使用默认配置文件
        default_config_path = "configs/Conv3D_train.yaml"
        if os.path.exists(default_config_path):
            config_loader = ConfigLoader(default_config_path)
        else:
            print(f"Warning: Config file {default_config_path} not found, using default config")
            config_loader = ConfigLoader()
    
    # 创建配置对象
    config = Config()
    
    # 从ConfigLoader更新配置
    training_config = config_loader.get_training_config()
    data_config = config_loader.get_data_config()
    checkpoint_config = config_loader.get_checkpoint_config()
    
    # 更新训练配置
    if 'epochs' in training_config:
        config.epochs = training_config['epochs']
    if 'batch_size' in training_config:
        config.batch_size = training_config['batch_size']
    if 'learning_rate' in training_config:
        config.learning_rate = training_config['learning_rate']
    
    # 更新数据配置
    if 'data_path' in data_config:
        config.data_path = data_config['data_path']
    if 'action_name_path' in data_config:
        config.action_name_path = data_config['action_name_path']
    if 'num_frames' in data_config:
        config.end_frame = data_config['num_frames'] + 1  # end_frame = num_frames + 1
    if 'begin_frame' in data_config:
        config.begin_frame = data_config['begin_frame']
    if 'end_frame' in data_config:
        config.end_frame = data_config['end_frame']
    if 'skip_frame' in data_config:
        config.skip_frame = data_config['skip_frame']
    
    # 更新检查点配置
    if 'save_dir' in checkpoint_config:
        config.save_model_path = checkpoint_config['save_dir']
    
    # 命令行参数覆盖
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.data_path is not None:
        config.data_path = args.data_path
    
    print("Configuration:")
    if not config.data_path or not config.action_name_path:
        raise ValueError("Missing required data paths. Please set data.data_path and data.action_name_path in config or CLI.")
    print(f"  Data path: {config.data_path}")
    print(f"  Action name path: {config.action_name_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # 创建保存目录
    os.makedirs(config.save_model_path, exist_ok=True)
    
    # 训练模型
    model = train_model(config)
    
    print("Conv3D training completed successfully!")

if __name__ == "__main__":
    main()

