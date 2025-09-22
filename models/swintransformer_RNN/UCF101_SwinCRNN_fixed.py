import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import logging
import sys
from datetime import datetime
from utils.common.model_components import SwinTransformerEncoder, DecoderRNN
from utils.common.data_loaders import Dataset_SwinCRNN
from utils.common.label_utils import labels2cat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import glob
from tqdm import tqdm
import time
import warnings
import yaml

# 过滤sklearn的警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

def setup_logging(log_dir):
    """设置日志记录"""
    # 处理相对路径和绝对路径
    if not os.path.isabs(log_dir):
        # 如果是相对路径，相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(project_root, log_dir)
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"swintransformer_rnn_training_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

class Config:
    def __init__(self, config_dict=None):
        # 设置默认值（这些值会在配置文件加载后被覆盖）
        self.data_path = None
        self.action_name_path = None
        self.save_model_path = None
        self.log_dir = None
        self.epochs = 50
        self.batch_size = 280
        self.learning_rate = 4e-3
        self.res_size = 224
        # SwinTransformer architecture
        self.CNN_fc_hidden1 = 1024
        self.CNN_fc_hidden2 = 768
        self.CNN_embed_dim = 512
        self.dropout_p = 0.0
        self.use_pretrained = True  # 是否使用预训练权重
        # DecoderRNN architecture
        self.RNN_hidden_layers = 3
        self.RNN_hidden_nodes = 512
        self.RNN_FC_dim = 256
        # training parameters
        self.k = 101
        self.log_interval = 10
        # Select which frame to begin & end in videos
        self.begin_frame = 1
        self.end_frame = 28
        self.skip_frame = 1
        
        # 如果提供了配置文件，则加载配置
        if config_dict:
            self._update_from_config(config_dict)
        else:
            raise ValueError("Configuration dictionary is required. Please provide a valid config file.")
    
    def _update_from_config(self, config_dict):
        """从配置文件更新配置"""
        # 数据配置
        if 'data' in config_dict:
            data_cfg = config_dict['data']
            self.data_path = data_cfg.get('data_path', self.data_path)
            self.action_name_path = data_cfg.get('action_name_path', self.action_name_path)
            self.k = int(data_cfg.get('num_classes', self.k))
            self.res_size = int(data_cfg.get('frame_size', self.res_size))
        
        # 训练配置
        if 'training' in config_dict:
            train_cfg = config_dict['training']
            self.epochs = int(train_cfg.get('epochs', self.epochs))
            self.batch_size = int(train_cfg.get('batch_size', self.batch_size))
            self.learning_rate = float(train_cfg.get('learning_rate', self.learning_rate))
        
        # 检查点配置
        if 'checkpoint' in config_dict:
            checkpoint_cfg = config_dict['checkpoint']
            self.save_model_path = str(checkpoint_cfg.get('save_dir', self.save_model_path))
            self.use_pretrained = bool(checkpoint_cfg.get('pretrained', self.use_pretrained))
        
        # 日志配置
        if 'logging' in config_dict:
            logging_cfg = config_dict['logging']
            self.log_dir = str(logging_cfg.get('log_dir', self.log_dir))
        
        # 验证必要的配置是否存在
        if self.data_path is None:
            raise ValueError("data_path is required in configuration")
        if self.action_name_path is None:
            raise ValueError("action_name_path is required in configuration")
        if self.save_model_path is None:
            raise ValueError("save_dir is required in checkpoint configuration")
        if self.log_dir is None:
            raise ValueError("log_dir is required in logging configuration")

def train_epoch(model, device, train_loader, optimizer, epoch, config):
    """训练一个epoch"""
    swin_encoder, rnn_decoder = model
    swin_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]', leave=False, ncols=100)
    
    for batch_idx, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device).view(-1, )

        optimizer.zero_grad()
        output = rnn_decoder(swin_encoder(X))

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        y_pred = torch.max(output, 1)[1]
        if len(y) > 1:
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        else:
            step_score = 1.0 if y_pred.item() == y.item() else 0.0
        scores.append(step_score)

        loss.backward()
        optimizer.step()

        avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
        avg_acc = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{avg_acc:.2%}',
            'Batch': f'{batch_idx+1}/{len(train_loader)}'
        })

    return losses, scores

def validate_epoch(model, device, val_loader, epoch, config, logger=None):
    """验证一个epoch"""
    swin_encoder, rnn_decoder = model
    swin_encoder.eval()
    rnn_decoder.eval()

    val_loss = 0
    all_y = []
    all_y_pred = []
    
    pbar = tqdm(val_loader, desc='[Validation]', leave=False, ncols=100)
    
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(swin_encoder(X))
            loss = F.cross_entropy(output, y, reduction='sum')
            val_loss += loss.item()
            
            y_pred = output.max(1, keepdim=True)[1]
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    val_loss /= len(val_loader.dataset)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    
    if len(all_y) > 1:
        val_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    else:
        val_score = 1.0 if all_y_pred.item() == all_y.item() else 0.0

    if logger:
        logger.info(f'\nValidation set ({len(all_y)} samples): Average loss: {val_loss:.4f}, Accuracy: {val_score:.2%}')
    else:
        print(f'\nValidation set ({len(all_y)} samples): Average loss: {val_loss:.4f}, Accuracy: {val_score:.2%}')

    return val_loss, val_score

def create_model(config):
    """创建SwinTransformer-RNN模型"""
    swin_encoder = SwinTransformerEncoder(
        fc_hidden1=config.CNN_fc_hidden1, 
        fc_hidden2=config.CNN_fc_hidden2, 
        drop_p=config.dropout_p, 
        CNN_embed_dim=config.CNN_embed_dim, 
        swin_model_name='swin_tiny_patch4_window7_224',
        use_pretrained=config.use_pretrained
    )
    rnn_decoder = DecoderRNN(
        CNN_embed_dim=config.CNN_embed_dim, 
        h_RNN_layers=config.RNN_hidden_layers, 
        h_RNN=config.RNN_hidden_nodes, 
        h_FC_dim=config.RNN_FC_dim, 
        drop_p=config.dropout_p, 
        num_classes=config.k
    )
    
    return swin_encoder, rnn_decoder

def get_data_loaders(config):
    """获取数据加载器"""
    # 加载动作名称
    with open(config.action_name_path, 'rb') as f:
        action_names = pickle.load(f)
    
    # 创建标签编码器
    le = LabelEncoder()
    le.fit(action_names)
    
    # 收集视频文件
    all_video_folders = []
    all_video_labels = []
    
    for action_name in action_names:
        action_path = os.path.join(config.data_path, action_name)
        if os.path.exists(action_path):
            video_folders = glob.glob(os.path.join(action_path, "v_*"))
            for video_folder in video_folders:
                frame_files = glob.glob(os.path.join(video_folder, "frame*.jpg"))
                if len(frame_files) >= config.end_frame:
                    all_video_folders.append(video_folder)
                    all_video_labels.append(action_name)
    
    # 转换为类别标签
    all_y_list = labels2cat(le, all_video_labels)
    
    # 数据分割
    X = np.array(all_video_folders)
    y = np.array(all_y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize([config.res_size, config.res_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    selected_frames = list(range(config.begin_frame, config.end_frame, config.skip_frame))
    
    train_dataset = Dataset_SwinCRNN(config.data_path, X_train, y_train, selected_frames, transform=transform)
    val_dataset = Dataset_SwinCRNN(config.data_path, X_val, y_val, selected_frames, transform=transform)
    
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, len(action_names)

def train_model(config):
    """训练模型的主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建保存目录
    os.makedirs(config.save_model_path, exist_ok=True)
    
    # 设置日志记录
    logger = setup_logging(config.log_dir)
    
    logger.info(f"Using device: {device}")
    
    # 获取数据加载器
    train_loader, val_loader, num_classes = get_data_loaders(config)
    logger.info(f"Data loaded: {num_classes} classes")
    
    # 创建模型
    swin_encoder, rnn_decoder = create_model(config)
    swin_encoder = swin_encoder.to(device)
    rnn_decoder = rnn_decoder.to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        swin_encoder = nn.DataParallel(swin_encoder)
        rnn_decoder = nn.DataParallel(rnn_decoder)
    
    model = [swin_encoder, rnn_decoder]
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        list(swin_encoder.parameters()) + list(rnn_decoder.parameters()), 
        lr=config.learning_rate
    )
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        # 训练
        train_losses, train_scores = train_epoch(model, device, train_loader, optimizer, epoch, config)
        
        # 验证
        val_loss, val_score = validate_epoch(model, device, val_loader, epoch, config, logger)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'swin_encoder_state_dict': swin_encoder.state_dict(),
                'rnn_decoder_state_dict': rnn_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_score': val_score
            }, os.path.join(config.save_model_path, 'best_model.pth'))
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_score:.2%}")
    
    logger.info("Training completed!")
    return model

if __name__ == '__main__':
    # 当直接运行此文件时，使用默认配置
    config = Config()
    train_model(config) 