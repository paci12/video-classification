#!/usr/bin/env python3
"""
Video Classification Training Script
支持多种模型的训练：Conv3D, CRNN, ResNetCRNN, ResNetCRNN_varylength, SwinTransformer-RNN
使用configs目录中的配置文件
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import pickle
import glob
from sklearn.preprocessing import LabelEncoder
import torch.utils.data as data

# 添加utils路径
sys.path.append('utils')

# 导入公共组件
from utils.common.label_utils import labels2cat, labels2onehot, onehot2labels, cat2labels
from utils.common.data_loaders import Dataset_CRNN, Dataset_3DCNN, Dataset_CRNN_varlen, Dataset_SwinCRNN
from utils.common.model_components import EncoderCNN, ResCNNEncoder, DecoderRNN
from utils.common.training_utils import (
    train_epoch, validate_epoch, save_checkpoint, load_checkpoint,
    get_optimizer, get_scheduler, save_predictions, count_parameters
)
from utils.config_loader import ConfigLoader
import torchvision.transforms as transforms

def load_config(model_name, config_path=None):
    """加载配置文件"""
    if config_path and os.path.exists(config_path):
        config_loader = ConfigLoader(config_path)
    else:
        # 使用默认配置文件
        default_config_path = f"configs/{model_name}_train.yaml"
        if os.path.exists(default_config_path):
            config_loader = ConfigLoader(default_config_path)
        else:
            print(f"Warning: Config file {default_config_path} not found, using default config")
            config_loader = ConfigLoader()
    
    return config_loader

def create_model(model_name, config):
    """根据模型名称和配置创建模型"""
    model_config = config.get_model_config()
    training_config = config.get_training_config()
    
    if model_name == 'Conv3D':
        from models.Conv3D.UCF101_3DCNN_refactored import Conv3DModel, Config
        
        # 创建Config实例，传入配置文件
        config_dict = config.config
        conv3d_config = Config()
        
        # 从配置文件更新配置
        if 'training' in config_dict:
            training = config_dict['training']
            if 'epochs' in training:
                conv3d_config.epochs = int(training['epochs'])
            if 'batch_size' in training:
                conv3d_config.batch_size = int(training['batch_size'])
            if 'learning_rate' in training:
                conv3d_config.learning_rate = float(training['learning_rate'])
        
        if 'data' in config_dict:
            data = config_dict['data']
            if 'data_path' in data:
                conv3d_config.data_path = str(data['data_path'])
            if 'action_name_path' in data:
                conv3d_config.action_name_path = str(data['action_name_path'])
            if 'num_classes' in data:
                conv3d_config.num_classes = int(data['num_classes'])
        
        if 'checkpoint' in config_dict:
            checkpoint = config_dict['checkpoint']
            if 'save_dir' in checkpoint:
                conv3d_config.save_model_path = str(checkpoint['save_dir'])
        
        return Conv3DModel(conv3d_config)
    
    elif model_name == 'CRNN':
        from models.CRNN.UCF101_CRNN_refactored import CRNNModel, Config
        
        # 创建Config实例，传入配置文件
        config_dict = config.config
        crnn_config = Config()
        
        # 从配置文件更新配置
        if 'training' in config_dict:
            training = config_dict['training']
            if 'epochs' in training:
                crnn_config.epochs = int(training['epochs'])
            if 'batch_size' in training:
                crnn_config.batch_size = int(training['batch_size'])
            if 'learning_rate' in training:
                crnn_config.learning_rate = float(training['learning_rate'])
        
        if 'data' in config_dict:
            data = config_dict['data']
            if 'data_path' in data:
                crnn_config.data_path = str(data['data_path'])
            if 'action_name_path' in data:
                crnn_config.action_name_path = str(data['action_name_path'])
            if 'num_classes' in data:
                crnn_config.num_classes = int(data['num_classes'])
            # 从配置文件读取图像尺寸
            if 'frame_size' in data:
                frame_size = int(data['frame_size'])
                crnn_config.img_x = frame_size
                crnn_config.img_y = frame_size
        
        # 从配置文件读取pretrained参数
        if 'checkpoint' in config_dict and 'pretrained' in config_dict['checkpoint']:
            crnn_config.use_pretrained = bool(config_dict['checkpoint']['pretrained'])
            print(f"CRNN: Using pretrained weights: {crnn_config.use_pretrained}")
        
        if 'checkpoint' in config_dict:
            checkpoint = config_dict['checkpoint']
            if 'save_dir' in checkpoint:
                crnn_config.save_model_path = str(checkpoint['save_dir'])
        
        return CRNNModel(crnn_config)
    
    elif model_name == 'ResNetCRNN':
        from models.ResNetCRNN.UCF101_ResNetCRNN_refactored import ResNetCRNNModel, Config
        
        # 创建Config实例，传入配置文件
        config_dict = config.config
        resnet_config = Config()
        
        # 从配置文件更新配置
        if 'training' in config_dict:
            training = config_dict['training']
            if 'epochs' in training:
                resnet_config.epochs = int(training['epochs'])
            if 'batch_size' in training:
                resnet_config.batch_size = int(training['batch_size'])
            if 'learning_rate' in training:
                resnet_config.learning_rate = float(training['learning_rate'])
        
        if 'data' in config_dict:
            data = config_dict['data']
            if 'data_path' in data:
                resnet_config.data_path = str(data['data_path'])
            if 'action_name_path' in data:
                resnet_config.action_name_path = str(data['action_name_path'])
            if 'num_classes' in data:
                resnet_config.num_classes = int(data['num_classes'])
        
        # 从配置文件读取pretrained参数
        if 'checkpoint' in config_dict and 'pretrained' in config_dict['checkpoint']:
            resnet_config.use_pretrained = bool(config_dict['checkpoint']['pretrained'])
            print(f"ResNetCRNN: Using pretrained weights: {resnet_config.use_pretrained}")
        
        if 'checkpoint' in config_dict:
            checkpoint = config_dict['checkpoint']
            if 'save_dir' in checkpoint:
                resnet_config.save_model_path = str(checkpoint['save_dir'])
        
        if 'model' in config_dict:
            model = config_dict['model']
            if 'fc_hidden1' in model:
                resnet_config.CNN_fc_hidden1 = int(model['fc_hidden1'])
            if 'fc_hidden2' in model:
                resnet_config.CNN_fc_hidden2 = int(model['fc_hidden2'])
            if 'embed_dim' in model:
                resnet_config.CNN_embed_dim = int(model['embed_dim'])
            if 'dropout' in model:
                resnet_config.dropout_p = float(model['dropout'])
            if 'hidden_layers' in model:
                resnet_config.RNN_hidden_layers = int(model['hidden_layers'])
            if 'hidden_nodes' in model:
                resnet_config.RNN_hidden_nodes = int(model['hidden_nodes'])
            if 'fc_dim' in model:
                resnet_config.RNN_FC_dim = int(model['fc_dim'])
        
        return ResNetCRNNModel(resnet_config)
    
    elif model_name == 'ResNetCRNN_varylength':
        # 使用可变长度的ResNetCRNN
        from models.ResNetCRNN_varylength.UCF101_ResNetCRNN_varlen import ResNetCRNNModel, Config
        
        # 创建Config实例，传入配置文件
        config_dict = config.config
        resnet_varlen_config = Config()
        
        # 从配置文件更新配置
        if 'training' in config_dict:
            training = config_dict['training']
            if 'epochs' in training:
                resnet_varlen_config.epochs = int(training['epochs'])
            if 'batch_size' in training:
                resnet_varlen_config.batch_size = int(training['batch_size'])
            if 'learning_rate' in training:
                resnet_varlen_config.learning_rate = float(training['learning_rate'])
        
        if 'data' in config_dict:
            data = config_dict['data']
            if 'data_path' in data:
                resnet_varlen_config.data_path = str(data['data_path'])
            if 'action_name_path' in data:
                resnet_varlen_config.action_name_path = str(data['action_name_path'])
            if 'num_classes' in data:
                resnet_varlen_config.num_classes = int(data['num_classes'])
            # 从配置文件读取图像尺寸
            if 'frame_size' in data:
                frame_size = int(data['frame_size'])
                resnet_varlen_config.img_x = frame_size
                resnet_varlen_config.img_y = frame_size
        
        if 'checkpoint' in config_dict:
            checkpoint = config_dict['checkpoint']
            if 'save_dir' in checkpoint:
                resnet_varlen_config.save_model_path = str(checkpoint['save_dir'])
        
        return ResNetCRNNModel(resnet_varlen_config)
    
    elif model_name == 'swintransformer-RNN':
        # 使用SwinTransformer-RNN
        from models.swintransformer_RNN.UCF101_SwinCRNN_fixed import create_model, Config
        
        # 创建Config实例，传入配置文件
        config_dict = config.config
        swin_config = Config(config_dict)
        
        # 创建模型
        swin_encoder, rnn_decoder = create_model(swin_config)
        
        class SwinTransformerRNNModel(nn.Module):
            def __init__(self, encoder, decoder):
                super(SwinTransformerRNNModel, self).__init__()
                self.encoder = encoder
                self.decoder = decoder
            
            def forward(self, x):
                cnn_out = self.encoder(x)
                output = self.decoder(cnn_out)
                return output
        
        return SwinTransformerRNNModel(swin_encoder, rnn_decoder)
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def get_data_loader(model_name, config, split='train'):
    """获取数据加载器"""
    print(f"  Getting data loader for {model_name}, split: {split}")
    
    data_config = config.get_data_config()
    augmentation_config = config.get_augmentation_config()
    training_config = config.get_training_config()
    
    print(f"  Data config: {data_config}")
    print(f"  Augmentation config: {augmentation_config}")
    print(f"  Training config: {training_config}")
    
    # 数据路径
    data_path = data_config.get('data_path', 'jpegs_256')
    print(f"  Data path: {data_path}")
    
    # 数据变换
    if model_name == 'Conv3D':
        # 3D CNN使用灰度图像
        transform = transforms.Compose([
            transforms.Resize([data_config.get('frame_size', 112), data_config.get('frame_size', 112)]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        DatasetClass = Dataset_3DCNN
    else:
        # 其他模型使用彩色图像
        transform = transforms.Compose([
            transforms.Resize([data_config.get('frame_size', 224), data_config.get('frame_size', 224)]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=augmentation_config.get('normalize', {}).get('mean', [0.485, 0.456, 0.406]),
                std=augmentation_config.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
            )
        ])
        if model_name == 'ResNetCRNN_varylength':
            DatasetClass = Dataset_CRNN_varlen
        else:
            DatasetClass = Dataset_CRNN
    
    # 帧选择
    num_frames = int(data_config.get('num_frames', 16))
    frames = list(range(1, num_frames + 1))
    print(f"  Num frames: {num_frames}, frames: {frames}")
    
    # 实现数据加载逻辑
    try:
        print(f"  Starting data loading logic...")
        
        # 加载动作名称
        action_name_path = data_config.get('action_name_path', 'data/UCF101actions.pkl')
        print(f"  Action name path: {action_name_path}")
        
        if not os.path.exists(action_name_path):
            print(f"Warning: Action names file {action_name_path} not found")
            return None
            
        with open(action_name_path, 'rb') as f:
            action_names = pickle.load(f)
        print(f"  Loaded {len(action_names)} action names")
        
        # 收集视频文件
        all_video_folders = []
        all_video_labels = []
        
        print(f"  Searching for videos in {data_path}")
        print(f"  Available actions: {action_names[:5]}...")  # 只显示前5个
        
        for action_name in action_names:
            action_path = os.path.join(data_path, action_name)
            if os.path.exists(action_path):
                video_folders = glob.glob(os.path.join(action_path, "v_*"))
                print(f"  Found {len(video_folders)} videos in {action_name}")
                for video_folder in video_folders:
                    frame_files = glob.glob(os.path.join(video_folder, "frame*.jpg"))
                    print(f"    Video {video_folder}: {len(frame_files)} frames, checking >= {num_frames}")
                    if len(frame_files) >= num_frames:
                        all_video_folders.append(video_folder)
                        all_video_labels.append(action_name)
        
        print(f"  Total valid videos found: {len(all_video_folders)}")
        
        if len(all_video_folders) == 0:
            print(f"Warning: No video files found in {data_path}")
            return None
        
        # 创建标签编码器
        from utils.common.label_utils import labels2cat
        le = LabelEncoder()
        le.fit(action_names)
        all_y_list = labels2cat(le, all_video_labels)
        
        # 数据分割
        from sklearn.model_selection import train_test_split
        X = np.array(all_video_folders)
        y = np.array(all_y_list)
        
        if split == 'train':
            X_data, _, y_data, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:  # val
            _, X_data, _, y_data = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 创建数据集
        dataset = DatasetClass(data_path, X_data, y_data, frames, transform=transform)
        
        # 创建数据加载器
        batch_size = training_config.get('batch_size', 32)
        # 在多GPU情况下，确保每个GPU至少获得2个样本
        if torch.cuda.device_count() > 1:
            min_batch_size = 2 * torch.cuda.device_count()
            if batch_size < min_batch_size:
                print(f"Warning: Batch size {batch_size} is too small for {torch.cuda.device_count()} GPUs. "
                      f"Each GPU needs at least 2 samples. Increasing batch size to {min_batch_size}")
                batch_size = min_batch_size
        
        loader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(split == 'train'),
            num_workers=4, 
            pin_memory=True
        )
        
        return loader
        
    except Exception as e:
        print(f"Error creating data loader: {e}")
        return None

def train_model(model_name, config, args):
    """训练模型"""
    print(f"Starting training for {model_name}...")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    print("Creating model...")
    try:
        model = create_model(model_name, config)
        print("Model created successfully")
        model = model.to(device)
        print("Model moved to device")
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            print("DataParallel applied")
        
        print(f"Model parameters: {count_parameters(model):,}")
        
        # 打印模型结构
        print("\nModel structure:")
        print("="*40)
        print(model)
        print("="*40)
        
    except Exception as e:
        print(f"Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 获取配置
    training_config = config.get_training_config()
    checkpoint_config = config.get_checkpoint_config()
    
    # 创建优化器和调度器
    optimizer = get_optimizer(
        model,
        optimizer_name=training_config.get('optimizer', 'adam'),
        lr=float(training_config.get('learning_rate', 0.001)),
        weight_decay=float(training_config.get('weight_decay', 1e-4))
    )
    
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=training_config.get('scheduler', 'step'),
        step_size=int(training_config.get('step_size', 20)),
        gamma=float(training_config.get('gamma', 0.1))
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 获取数据加载器
    print("Creating data loaders...")
    try:
        print("Creating train loader...")
        train_loader = get_data_loader(model_name, config, 'train')
        print(f"Train loader created: {train_loader is not None}")
        
        print("Creating val loader...")
        val_loader = get_data_loader(model_name, config, 'val')
        print(f"Val loader created: {val_loader is not None}")
        
        if train_loader is None or val_loader is None:
            print("Error: Data loading not implemented. Please use the individual model scripts.")
            return
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 训练循环
    epochs = args.epochs or training_config.get('epochs', 50)
    best_val_loss = float('inf')
    
    # 初始化训练历史记录
    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # 验证
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, epoch+1
        )
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存训练历史
        epoch_train_losses.append(train_loss)
        epoch_train_scores.append(train_acc)
        epoch_test_losses.append(val_loss)
        epoch_test_scores.append(val_acc)
        
        # 实时保存训练历史到文件
        base_dir = checkpoint_config.get('save_dir', f'results/{model_name}')
        outputs_dir = os.path.join(base_dir, 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        
        # 保存当前epoch的训练历史
        np.save(os.path.join(outputs_dir, f'{model_name}_epoch_training_losses.npy'), np.array(epoch_train_losses))
        np.save(os.path.join(outputs_dir, f'{model_name}_epoch_training_scores.npy'), np.array(epoch_train_scores))
        np.save(os.path.join(outputs_dir, f'{model_name}_epoch_test_loss.npy'), np.array(epoch_test_losses))
        np.save(os.path.join(outputs_dir, f'{model_name}_epoch_test_score.npy'), np.array(epoch_test_scores))
        
        # 保存训练日志
        log_file = os.path.join(outputs_dir, f'{model_name}_training_log.txt')
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
        
        # 实时生成训练曲线图
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 损失曲线
            ax1.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses, label='Train Loss', marker='o')
            ax1.plot(range(1, len(epoch_test_losses) + 1), epoch_test_losses, label='Val Loss', marker='s')
            ax1.set_title(f'{model_name} Training Loss (Epoch {epoch+1})')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # 准确率曲线
            ax2.plot(range(1, len(epoch_train_scores) + 1), epoch_train_scores, label='Train Acc', marker='o')
            ax2.plot(range(1, len(epoch_test_scores) + 1), epoch_test_scores, label='Val Acc', marker='s')
            ax2.set_title(f'{model_name} Training Accuracy (Epoch {epoch+1})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(outputs_dir, f'{model_name}_training_curves_latest.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("matplotlib 未安装，跳过训练曲线图生成")
        
        # 保存训练进度为JSON格式
        try:
            import json
            
            progress_data = {
                'epoch': epoch + 1,
                'total_epochs': epochs,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
                'training_history': {
                    'train_losses': epoch_train_losses,
                    'train_scores': epoch_train_scores,
                    'val_losses': epoch_test_losses,
                    'val_scores': epoch_test_scores
                }
            }
            
            progress_file = os.path.join(outputs_dir, f'{model_name}_training_progress.json')
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save progress JSON: {e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 从配置文件读取基础保存目录，然后拼接权重子目录
            base_dir = checkpoint_config.get('save_dir', f'results/{model_name}')
            if model_name == 'swintransformer-RNN':
                ckpt_dir = os.path.join(base_dir, 'SwinCRNN_ckpt')
            elif model_name == 'ResNetCRNN':
                ckpt_dir = os.path.join(base_dir, 'ResNetCRNN_ckpt')
            elif model_name == 'CRNN':
                ckpt_dir = os.path.join(base_dir, 'CRNN_ckpt')
            elif model_name == 'Conv3D':
                ckpt_dir = os.path.join(base_dir, 'Conv3D_ckpt')
            else:
                ckpt_dir = os.path.join(base_dir, 'outputs')
            
            save_path = os.path.join(ckpt_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch+1, val_loss, save_path)
        
        # 定期保存检查点
        save_freq = checkpoint_config.get('save_freq', 10)
        if (epoch + 1) % save_freq == 0:
            # 从配置文件读取基础保存目录，然后拼接权重子目录
            base_dir = checkpoint_config.get('save_dir', f'results/{model_name}')
            if model_name == 'swintransformer-RNN':
                ckpt_dir = os.path.join(base_dir, 'SwinCRNN_ckpt')
            elif model_name == 'ResNetCRNN':
                ckpt_dir = os.path.join(base_dir, 'ResNetCRNN_ckpt')
            elif model_name == 'CRNN':
                ckpt_dir = os.path.join(base_dir, 'CRNN_ckpt')
            elif model_name == 'Conv3D':
                ckpt_dir = os.path.join(base_dir, 'Conv3D_ckpt')
            else:
                ckpt_dir = os.path.join(base_dir, 'outputs')
            
            save_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch+1, val_loss, save_path)
        
        scheduler.step()
    
    print("Training completed!")
    
    # 保存训练历史到 outputs 目录
    base_dir = checkpoint_config.get('save_dir', f'results/{model_name}')
    outputs_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # 保存训练历史为 .npy 文件
    np.save(os.path.join(outputs_dir, f'{model_name}_epoch_training_losses.npy'), np.array(epoch_train_losses))
    np.save(os.path.join(outputs_dir, f'{model_name}_epoch_training_scores.npy'), np.array(epoch_train_scores))
    np.save(os.path.join(outputs_dir, f'{model_name}_epoch_test_loss.npy'), np.array(epoch_test_losses))
    np.save(os.path.join(outputs_dir, f'{model_name}_epoch_test_score.npy'), np.array(epoch_test_scores))
    
    # 生成训练曲线图
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses, label='Train Loss')
        ax1.plot(range(1, len(epoch_test_losses) + 1), epoch_test_losses, label='Val Loss')
        ax1.set_title(f'{model_name} Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(range(1, len(epoch_train_scores) + 1), epoch_train_scores, label='Train Acc')
        ax2.plot(range(1, len(epoch_test_scores) + 1), epoch_test_scores, label='Val Acc')
        ax2.set_title(f'{model_name} Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, f'{model_name}_training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练曲线图已保存到: {outputs_dir}/{model_name}_training_curves.png")
        
    except ImportError:
        print("matplotlib 未安装，跳过训练曲线图生成")
    
    # 保存预测结果到 check_predictions 目录
    check_predictions_dir = os.path.join(
        checkpoint_config.get('save_dir', f'results/{model_name}'),
        'check_predictions'
    )
    os.makedirs(check_predictions_dir, exist_ok=True)
    
    predictions_path = os.path.join(
        check_predictions_dir,
        f'{model_name}_epoch_latest_videos_prediction.pkl'
    )
    save_predictions(model, val_loader, device, predictions_path)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Video Classification Training')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['Conv3D', 'CRNN', 'ResNetCRNN', 'ResNetCRNN_varylength', 'swintransformer-RNN'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (optional, will use default config if not provided)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data directory (overrides config)')
    parser.add_argument('--resume', type=int, default=None,
                       help='Resume training from epoch (e.g., --resume 45 to resume from epoch 45)')
    parser.add_argument('--no_backup', action='store_true',
                       help='Do not backup existing result directory (useful for resuming)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.model, args.config)
    
    # 打印配置信息
    print("="*60)
    print(f"Training {args.model} model")
    print("="*60)
    
    # 添加调试信息
    print(f"Config object type: {type(config)}")
    print(f"Config object: {config}")
    print(f"Config config attribute: {getattr(config, 'config', 'No config attribute')}")
    
    config.print_config()
    
    # 更新配置（如果命令行参数提供了）
    updates = {}
    if args.epochs is not None:
        updates['training'] = {'epochs': args.epochs}
    if args.batch_size is not None:
        updates['training'] = updates.get('training', {})
        updates['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        updates['training'] = updates.get('training', {})
        updates['training']['learning_rate'] = args.lr
    if args.data_path is not None:
        updates['data'] = {'data_path': args.data_path}
    
    if updates:
        config.update_config(updates)
        print("\nUpdated configuration:")
        config.print_config()
    
    # 备份原有的result目录并创建新的result目录
    try:
        from datetime import datetime
        import shutil
        
        # 获取基础保存目录
        base_save_dir = config.get_checkpoint_config().get('save_dir', f'results/{args.model}/result')
        base_dir = os.path.dirname(base_save_dir)  # results/{model_name}
        result_dir = base_save_dir  # results/{model_name}/result
        
        print(f"\n检查目录: {result_dir}")
        
        # 如果使用resume模式，不备份目录
        if args.resume is not None:
            print(f"🔄 Resume模式：从epoch {args.resume}继续训练")
            if os.path.exists(result_dir):
                print(f"✅ 使用现有目录: {result_dir}")
                # 检查checkpoint文件是否存在
                ckpt_dir = os.path.join(result_dir, f"{args.model}_ckpt")
                if os.path.exists(ckpt_dir):
                    checkpoint_files = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_epoch_')]
                    print(f"📁 发现checkpoint文件: {checkpoint_files}")
                else:
                    print(f"⚠️  警告：checkpoint目录不存在: {ckpt_dir}")
            else:
                print(f"❌ 错误：resume模式需要现有目录，但 {result_dir} 不存在")
                print("请先运行一次训练以创建必要的目录结构")
                sys.exit(1)
        else:
            # 正常模式：备份旧目录
            if os.path.exists(result_dir):
                # 获取result目录的创建时间（或修改时间）
                try:
                    # 尝试获取目录的创建时间
                    stat_info = os.stat(result_dir)
                    # 使用修改时间作为备份目录名（更可靠）
                    dir_time = datetime.fromtimestamp(stat_info.st_mtime)
                    timestamp = dir_time.strftime("%Y%m%d_%H%M%S")
                    print(f"发现已存在的result目录，创建时间: {dir_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    # 如果获取时间失败，使用当前时间
                    print(f"无法获取目录时间，使用当前时间: {e}")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                backup_dir = f"{base_dir}/result_{timestamp}"
                print(f"正在备份到: {backup_dir}")
                
                # 备份目录
                shutil.move(result_dir, backup_dir)
                print(f"✅ 目录备份完成: {result_dir} -> {backup_dir}")
            
            # 创建新的result目录
            os.makedirs(result_dir, exist_ok=True)
            print(f"✅ 创建新的result目录: {result_dir}")
        
        # 创建必要的子目录（如果不存在）
        ckpt_dir = os.path.join(result_dir, f"{args.model}_ckpt")
        outputs_dir = os.path.join(result_dir, "outputs")
        check_predictions_dir = os.path.join(result_dir, "check_predictions")
        
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(check_predictions_dir, exist_ok=True)
        
        print(f"✅ 子目录状态:")
        print(f"   - 权重目录: {ckpt_dir}")
        print(f"   - 输出目录: {outputs_dir}")
        print(f"   - 预测结果目录: {check_predictions_dir}")
        
    except Exception as e:
        print(f"⚠️  目录处理过程中出现错误: {e}")
        print("继续训练，但可能无法正确保存结果")
        # 确保至少创建基本的result目录
        try:
            os.makedirs(base_save_dir, exist_ok=True)
        except:
            pass
    
    # 训练模型
    try:
        model = train_model(args.model, config, args)
        print(f"\n{args.model} training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        print("\nNote: For full functionality, please use the individual model scripts:")
        print(f"  python models/{args.model}/UCF101_{args.model}_refactored.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
