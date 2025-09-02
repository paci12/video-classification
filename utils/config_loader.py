#!/usr/bin/env python3
"""
Configuration Loader for Video Classification Models
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        return self.config
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get('training', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config.get('data', {})
    
    def get_augmentation_config(self) -> Dict[str, Any]:
        """获取数据增强配置"""
        return self.config.get('augmentation', {})
    
    def get_checkpoint_config(self) -> Dict[str, Any]:
        """获取检查点配置"""
        return self.config.get('checkpoint', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self.config.get('logging', {})
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 要更新的配置字典
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
    
    def save_config(self, save_path: str):
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def print_config(self):
        """打印配置信息"""
        print("="*60)
        print("Configuration:")
        print("="*60)
        try:
            config_str = yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
            print(config_str)
        except Exception as e:
            print(f"Error dumping config: {e}")
            print("Config content:")
            print(self.config)
        print("="*60)

def load_model_config(model_name: str) -> ConfigLoader:
    """
    根据模型名称加载对应的配置文件
    
    Args:
        model_name: 模型名称
        
    Returns:
        配置加载器实例
    """
    config_file = f"configs/{model_name}_train.yaml"
    if os.path.exists(config_file):
        return ConfigLoader(config_file)
    else:
        print(f"Warning: Config file {config_file} not found, using default config")
        return ConfigLoader()

def create_default_config(model_name: str) -> Dict[str, Any]:
    """
    创建默认配置
    
    Args:
        model_name: 模型名称
        
    Returns:
        默认配置字典
    """
    default_config = {
        'model': {
            'name': model_name,
            'num_classes': 101
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        },
        'data': {
            'dataset': 'UCF101',
            'data_path': 'jpegs_256',
            'num_frames': 16,
            'frame_size': 224
        },
        'checkpoint': {
            'save_dir': f'results/{model_name}/outputs',
            'save_freq': 5
        },
        'logging': {
            'log_dir': f'results/{model_name}/outputs/logs',
            'tensorboard': True
        }
    }
    return default_config

if __name__ == "__main__":
    # 测试配置加载器
    config_loader = ConfigLoader("configs/ResNetCRNN_train.yaml")
    config_loader.print_config()
    
    print("\nModel config:", config_loader.get_model_config())
    print("Training config:", config_loader.get_training_config())
