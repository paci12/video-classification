import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
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

class Config:
    def __init__(self, config_dict=None):
        # 默认值
        self.data_path = "jpegs_256_processed"
        self.action_name_path = 'configs/data/UCF101actions.pkl'
        self.save_model_path = "results/swintransformer_rnn/outputs"
        self.epochs = 120
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
        
        # 如果提供了配置文件，则覆盖默认值
        if config_dict:
            self._update_from_config(config_dict)
    
    def _update_from_config(self, config_dict):
        """从配置文件更新配置"""
        if 'data' in config_dict:
            data_cfg = config_dict['data']
            if 'data_path' in data_cfg:
                self.data_path = data_cfg['data_path']
            if 'action_name_path' in data_cfg:
                self.action_name_path = data_cfg['action_name_path']
            if 'num_classes' in data_cfg:
                self.k = int(data_cfg['num_classes'])
            if 'frame_size' in data_cfg:
                self.res_size = int(data_cfg['frame_size'])
        
        if 'training' in config_dict:
            train_cfg = config_dict['training']
            if 'epochs' in train_cfg:
                self.epochs = int(train_cfg['epochs'])
            if 'batch_size' in train_cfg:
                self.batch_size = int(train_cfg['batch_size'])
            if 'learning_rate' in train_cfg:
                self.learning_rate = float(train_cfg['learning_rate'])
        
        if 'checkpoint' in config_dict:
            checkpoint_cfg = config_dict['checkpoint']
            if 'save_dir' in checkpoint_cfg:
                self.save_model_path = str(checkpoint_cfg['save_dir'])
            # 读取pretrained参数
            if 'pretrained' in checkpoint_cfg:
                self.use_pretrained = bool(checkpoint_cfg['pretrained'])
                print(f"SwinTransformer: Using pretrained weights: {self.use_pretrained}")

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

def validate_epoch(model, device, val_loader, epoch, config):
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
    print(f"Using device: {device}")
    
    # 创建保存目录
    os.makedirs(config.save_model_path, exist_ok=True)
    
    # 获取数据加载器
    train_loader, val_loader, num_classes = get_data_loaders(config)
    print(f"Data loaded: {num_classes} classes")
    
    # 创建模型
    swin_encoder, rnn_decoder = create_model(config)
    swin_encoder = swin_encoder.to(device)
    rnn_decoder = rnn_decoder.to(device)
    
    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
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
        val_loss, val_score = validate_epoch(model, device, val_loader, epoch, config)
        
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
        
        print(f"Epoch {epoch+1}: Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_score:.2%}")
    
    print("Training completed!")
    return model

if __name__ == '__main__':
    # 当直接运行此文件时，使用默认配置
    config = Config()
    train_model(config) 