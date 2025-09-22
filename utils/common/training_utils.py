"""
Training utilities for video classification models
视频分类模型的训练工具
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Optional AMP
try:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_AVAILABLE = True
except Exception:  # pragma: no cover
    autocast = None
    GradScaler = None
    _AMP_AVAILABLE = False

def set_deterministic(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    """Seed dataloader workers deterministically."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

def get_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, use_amp: bool = False, scaler: Optional["GradScaler"] = None, logger=None):
    """
    训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        
    Returns:
        平均训练损失
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        if use_amp and _AMP_AVAILABLE:
            if scaler is None:
                scaler = GradScaler()
            with autocast():
                output = model(data)
                loss = criterion(output, target.squeeze())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target.squeeze())
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
        
        # 每100个batch记录一次日志
        if logger and (batch_idx + 1) % 100 == 0:
            current_loss = loss.item()
            current_acc = 100. * correct / total
            logger.info(f"Epoch {epoch} - Batch {batch_idx + 1}/{len(train_loader)} - Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device, epoch, use_amp: bool = False, logger=None):
    """
    验证一个epoch
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        
    Returns:
        平均验证损失和准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Validation')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            if use_amp and _AMP_AVAILABLE:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target.squeeze())
            else:
                output = model(data)
                loss = criterion(output, target.squeeze())
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # 每50个batch记录一次验证日志
            if logger and (batch_idx + 1) % 50 == 0:
                current_loss = loss.item()
                current_acc = 100. * correct / total
                logger.info(f"Epoch {epoch} - Val Batch {batch_idx + 1}/{len(val_loader)} - Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

class EarlyStopping:
    """Early stopping utility tracking a validation metric."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        assert mode in ('min', 'max')
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improvement = (value < self.best - self.min_delta) if self.mode == 'min' else (value > self.best + self.min_delta)
        if improvement:
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience

def is_better(curr: float, best: Optional[float], mode: str = 'min') -> bool:
    if best is None:
        return True
    return (curr < best) if mode == 'min' else (curr > best)

def save_best_checkpoint(model, optimizer, epoch, metric_value: float, best_metric: Optional[float], mode: str, save_path: str) -> float:
    """Save model if metric improves; returns updated best metric."""
    if is_better(metric_value, best_metric, mode):
        save_checkpoint(model, optimizer, epoch, metric_value, save_path)
        return metric_value
    return best_metric if best_metric is not None else metric_value

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    Returns:
        起始epoch
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from epoch 1")
        return 1

def save_predictions(model, test_loader, device, save_path, action_names=None):
    """
    保存预测结果
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        save_path: 保存路径
        action_names: 动作名称列表
    """
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Generating predictions'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            predictions.extend(pred.cpu().numpy().flatten())
            true_labels.extend(target.cpu().numpy().flatten())
    
    # 创建预测结果DataFrame
    results_df = pd.DataFrame({
        'y': true_labels,
        'y_pred': predictions
    })
    
    # 保存预测结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(results_df, f)
    
    print(f"Predictions saved to {save_path}")
    return results_df

def evaluate_predictions(y_true: List[int], y_pred: List[int], labels: Optional[List[int]] = None, target_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute accuracy, macro/micro F1, confusion matrix and classification report."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0, output_dict=True)
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm,
        'classification_report': report,
    }

def save_evaluation(metrics: Dict[str, Any], save_dir: str) -> None:
    """Persist evaluation metrics to disk (JSON/CSV + confusion matrix .npy)."""
    import json
    os.makedirs(save_dir, exist_ok=True)
    # Save summary JSON
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in metrics.items() if k != 'classification_report'}, f, ensure_ascii=False, indent=2)
    # Save classification report CSV
    report_df = pd.DataFrame(metrics['classification_report']).transpose()
    report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
    # Save confusion matrix
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), metrics['confusion_matrix'])

def get_optimizer(model, optimizer_name='adam', lr=0.001, weight_decay=1e-4):
    """
    获取优化器
    
    Args:
        model: 模型
        optimizer_name: 优化器名称 ('adam', 'sgd', 'adamw')
        lr: 学习率
        weight_decay: 权重衰减
        
    Returns:
        优化器
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name='step', step_size=20, gamma=0.1):
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 ('step', 'cosine', 'plateau')
        step_size: 步长
        gamma: 衰减因子
        
    Returns:
        学习率调度器
    """
    if scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    elif scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=step_size)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def count_parameters(model):
    """
    计算模型参数数量
    
    Args:
        model: 模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
