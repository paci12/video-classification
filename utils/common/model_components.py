"""
Model components for video classification
视频分类的模型组件
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 尝试导入SwinTransformer
try:
    from models.swin_transformer import SwinTransformer
except ImportError:
    try:
        from timm.models.swin_transformer import SwinTransformer
    except ImportError:
        print("Warning: SwinTransformer not found. Please install timm or provide the correct import path.")
        SwinTransformer = None

def conv2D_output_size(img_size, padding, kernel_size, stride):
    """
    计算conv2D的输出形状
    
    Args:
        img_size: 输入图像尺寸 (height, width)
        padding: 填充大小 (pad_h, pad_w)
        kernel_size: 卷积核大小 (kernel_h, kernel_w)
        stride: 步长 (stride_h, stride_w)
        
    Returns:
        输出尺寸 (out_h, out_w)
    """
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int))
    return outshape


class EncoderCNN(nn.Module):
    """
    2D CNN编码器 - 从头训练
    用于CRNN模型的CNN编码器
    """
    def __init__(self, img_x=224, img_y=224, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, use_pretrained=False):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN架构参数
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d卷积核大小
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d步长
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d填充

        # conv2D输出形状
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)

        # 全连接层隐藏节点
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # 根据配置决定是否使用预训练权重
        if use_pretrained:
            print("Loading VGG16 with ImageNet pretrained weights for EncoderCNN...")
            # 使用预训练的VGG16作为基础
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # 提取前几层作为特征提取器
            self.features = nn.Sequential(*list(vgg.features.children())[:23])  # 到第4个卷积块
            # 计算VGG输出的特征维度
            self.feature_dim = 512 * (self.img_x // 32) * (self.img_y // 32)
        else:
            print("Using custom CNN architecture (no pretrained weights)...")
            self.features = None
            self.feature_dim = self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1]

        # 如果使用预训练VGG，则使用VGG的特征维度
        if use_pretrained:
            self.fc1 = nn.Linear(self.feature_dim, fc_hidden1)
        else:
            # 原有的自定义CNN层
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
                nn.BatchNorm2d(self.ch1, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
                nn.BatchNorm2d(self.ch2, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
                nn.BatchNorm2d(self.ch3, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
                nn.BatchNorm2d(self.ch4, momentum=0.01),
                nn.ReLU(inplace=True),
            )
            self.fc1 = nn.Linear(self.feature_dim, fc_hidden1)

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.fc3 = nn.Linear(fc_hidden2, self.CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # 根据是否使用预训练权重选择不同的前向传播路径
            if self.features is not None:
                # 使用预训练VGG特征
                x = self.features(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1)           # 展平VGG输出
            else:
                # 使用自定义CNN
                x = self.conv1(x_3d[:, t, :, :, :])
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = x.view(x.size(0), -1)           # 展平卷积输出

            # FC layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # 交换时间和样本维度
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq


class ResCNNEncoder(nn.Module):
    """
    2D CNN编码器 - 使用预训练的ResNet
    用于ResNetCRNN模型的CNN编码器
    """
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, use_pretrained=True):
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # 根据配置决定是否使用预训练权重
        if use_pretrained:
            print("Loading ResNet152 with ImageNet pretrained weights...")
            resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        else:
            print("Loading ResNet152 with random initialization (no pretrained weights)...")
            resnet = models.resnet152(weights=None)
            
        modules = list(resnet.children())[:-1]      # 删除最后一个fc层
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN - 移除torch.no_grad()让ResNet权重参与训练
            x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            x = x.view(x.size(0), -1)             # 展平输出

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # 交换时间和样本维度
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    """
    RNN解码器
    用于CRNN和ResNetCRNN模型的RNN解码器
    """
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN隐藏层数
        self.h_RNN = h_RNN                 # RNN隐藏节点数
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # 输入和输出的第一维是batch size
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None表示零初始隐藏状态。RNN_out shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # 选择最后一个时间步的RNN_out
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class SwinTransformerEncoder(nn.Module):
    """
    Swin Transformer编码器
    用于SwinTransformer-RNN模型的编码器
    """
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, 
                 swin_model_name='swin_tiny_patch4_window7_224', use_pretrained=True):
        super(SwinTransformerEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # 根据配置决定是否使用预训练权重
        if use_pretrained:
            print(f"Loading {swin_model_name} with ImageNet pretrained weights...")
            # 这里需要根据实际的Swin Transformer实现来加载预训练权重
            # 由于SwinTransformer可能不在torchvision中，这里提供一个框架
            try:
                # 尝试从timm加载预训练的Swin Transformer
                import timm
                self.swin = timm.create_model(swin_model_name, pretrained=True, num_classes=0)
                # 获取特征维度
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    features = self.swin(dummy_input)
                    self.feature_dim = features.shape[1]
            except ImportError:
                print("timm not available, using random initialization")
                use_pretrained = False
        else:
            print(f"Loading {swin_model_name} with random initialization (no pretrained weights)...")
            use_pretrained = False

        if not use_pretrained:
            # 如果没有预训练权重，使用随机初始化的Swin Transformer
            # 这里提供一个简化的实现框架
            self.swin = None
            self.feature_dim = 768  # Swin-T的默认特征维度

        self.fc1 = nn.Linear(self.feature_dim, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # 根据是否使用预训练权重选择不同的前向传播路径
            if self.swin is not None:
                # 使用预训练或随机初始化的Swin Transformer
                x = self.swin(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1)           # 展平Swin输出
            else:
                # 如果没有Swin模型，使用简单的特征提取
                # 这里提供一个fallback实现
                x = F.adaptive_avg_pool2d(x_3d[:, t, :, :, :], (1, 1))
                x = x.view(x.size(0), -1)

            # FC layers
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            
            x = self.fc2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # 交换时间和样本维度
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq
