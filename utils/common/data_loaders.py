"""
Data loaders for video classification models
视频分类模型的数据加载器
"""

import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn.functional as F

class Dataset_3DCNN(data.Dataset):
    """
    3D CNN数据加载器
    用于Conv3D模型的数据加载
    """
    def __init__(self, data_path, folders, labels, frames, transform=None):
        """
        初始化
        
        Args:
            data_path: 数据路径
            folders: 文件夹列表
            labels: 标签列表
            frames: 帧列表
            transform: 数据变换
        """
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        """返回数据集大小"""
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        """读取图像序列"""
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        """获取单个样本"""
        folder = self.folders[index]
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)
        y = torch.LongTensor([self.labels[index]])

        return X, y


class Dataset_CRNN(data.Dataset):
    """
    CRNN数据加载器
    用于CRNN和ResNetCRNN模型的数据加载
    """
    def __init__(self, data_path, folders, labels, frames, transform=None):
        """
        初始化
        
        Args:
            data_path: 数据路径
            folders: 文件夹列表
            labels: 标签列表
            frames: 帧列表
            transform: 数据变换
        """
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        """返回数据集大小"""
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        """读取图像序列"""
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        """获取单个样本"""
        folder = self.folders[index]
        X = self.read_images(self.data_path, folder, self.transform)
        y = torch.LongTensor([self.labels[index]])

        return X, y


class Dataset_CRNN_varlen(data.Dataset):
    """
    可变长度CRNN数据加载器
    用于ResNetCRNN_varylength模型的数据加载
    """
    def __init__(self, data_path, folders, labels, frames, transform=None):
        """
        初始化
        
        Args:
            data_path: 数据路径
            folders: 文件夹列表
            labels: 标签列表
            frames: 帧列表
            transform: 数据变换
        """
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        """返回数据集大小"""
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        """读取图像序列"""
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        """获取单个样本"""
        folder = self.folders[index]
        X = self.read_images(self.data_path, folder, self.transform)
        y = torch.LongTensor([self.labels[index]])

        return X, y


class Dataset_SwinCRNN(data.Dataset):
    """
    SwinTransformer-CRNN数据加载器
    用于SwinTransformer-RNN模型的数据加载
    """
    def __init__(self, data_path, folders, labels, frames, transform=None):
        """
        初始化
        
        Args:
            data_path: 数据路径
            folders: 文件夹列表
            labels: 标签列表
            frames: 帧列表
            transform: 数据变换
        """
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        """返回数据集大小"""
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        """读取图像序列"""
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        """获取单个样本"""
        folder = self.folders[index]
        X = self.read_images(self.data_path, folder, self.transform)
        y = torch.LongTensor([self.labels[index]])

        return X, y
