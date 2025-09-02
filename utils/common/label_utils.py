"""
Label conversion utilities for video classification
视频分类的标签转换工具
"""

import numpy as np

def labels2cat(label_encoder, list):
    """
    将标签列表转换为类别编码
    
    Args:
        label_encoder: 标签编码器
        list: 标签列表
        
    Returns:
        类别编码数组
    """
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    """
    将标签列表转换为one-hot编码
    
    Args:
        OneHotEncoder: One-hot编码器
        label_encoder: 标签编码器
        list: 标签列表
        
    Returns:
        One-hot编码数组
    """
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    """
    将one-hot编码转换为标签列表
    
    Args:
        label_encoder: 标签编码器
        y_onehot: One-hot编码数组
        
    Returns:
        标签列表
    """
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    """
    将类别编码转换为标签列表
    
    Args:
        label_encoder: 标签编码器
        y_cat: 类别编码数组
        
    Returns:
        标签列表
    """
    return label_encoder.inverse_transform(y_cat).tolist()
