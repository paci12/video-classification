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


class OffsetHead(nn.Module):
    def __init__(self, in_ch, num_windows):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, 2 * num_windows, 1)  # 每个window预测 Δx, Δy

    def forward(self, x):
        o = self.pw(F.relu(self.dw(x)))  # [B, 2W, H', W']
        return o.tanh()  # 归一化到 [-1,1]，再按特征尺寸缩放


class ClusterSwinEncoder(nn.Module):
    def __init__(self, backbone='swin_tiny_patch4_window7_224', embed_dim=512, k_tokens=8):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=True, features_only=True, out_indices=(1, 2, 3))
        ch2, ch3, ch4 = self.backbone.feature_info.channels()
        self.proj2 = nn.Conv2d(ch2, embed_dim, 1)
        self.proj3 = nn.Conv2d(ch3, embed_dim, 1)
        self.proj4 = nn.Conv2d(ch4, embed_dim, 1)
        self.offset2 = OffsetHead(ch2, num_windows=((14 * 14) // 49))  # 示例
        self.offset3 = OffsetHead(ch3, num_windows=((7 * 7) // 49))
        self.offset4 = OffsetHead(ch4, num_windows=((7 * 7) // 49))
        self.k_tokens = k_tokens
        self.token_pool = nn.AdaptiveAvgPool2d(1)

    def _residual_cascade(self, f2, f3, f4):
        f3u = F.interpolate(self.proj3(f3), size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f4u = F.interpolate(self.proj4(f4), size=f2.shape[-2:], mode='bilinear', align_corners=False)
        f2p = self.proj2(f2)
        return f4u + f3u + f2p  # [B, embed, H2, W2]

    def _cluster_tokens(self, fmap):
        B, C, H, W = fmap.shape
        heat = fmap.abs().mean(1)  # [B, H, W]
        k = min(self.k_tokens, H * W)
        topk = torch.topk(heat.view(B, -1), k, dim=1).indices  # [B, k]
        tokens = []
        for b in range(B):
            idxs = topk[b]
            y = (idxs // W).long()
            x = (idxs % W).long()
            tokens.append(fmap[b, :, y, x].transpose(0, 1))  # [k, C]
        return torch.stack(tokens, 0)  # [B, k, C]

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats, tokens = [], []
        for t in range(T):
            f2, f3, f4 = self.backbone(x[:, t])  # timm: 输入 [B, 3, H, W]
            fmap = self._residual_cascade(f2, f3, f4)  # [B, embed, h, w]
            feats.append(self.token_pool(fmap).flatten(1))  # [B, embed]
            tokens.append(self._cluster_tokens(fmap))  # [B, k, embed]
        g = torch.stack(feats, dim=1)  # [B, T, embed]
        c = torch.stack(tokens, dim=1)  # [B, T, k, embed]
        return g, c  # 返回时空特征和团簇tokens


class TCG_LSTM(nn.Module):
    def __init__(self, embed_dim=512, hidden=512, layers=2, k_tokens=8, down=4, num_classes=4, drop=0.1):
        super().__init__()
        self.down = down
        self.lstm_short = nn.LSTM(embed_dim + embed_dim, hidden, layers, batch_first=True, dropout=drop, bidirectional=True)
        self.lstm_long = nn.LSTM(embed_dim, hidden // 2, layers, batch_first=True, dropout=drop, bidirectional=True)
        self.gate = nn.Sequential(nn.Linear(hidden * 2 + hidden, hidden * 2), nn.Sigmoid())  # gate short by long
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 256), 
            nn.ReLU(True), 
            nn.Dropout(drop), 
            nn.Linear(256, num_classes)
        )

    def forward(self, g, c):
        # g: [B, T, E] (全局特征); c: [B, T, K, E] (团簇tokens)
        c_mean = c.mean(2)  # [B, T, E] 作为团簇摘要
        short_in = torch.cat([g, c_mean], dim=-1)  # [B, T, 2E]
        long_in = g[:, ::self.down, :]  # [B, T/down, E]
        
        # 短程和长程LSTM
        h_s, _ = self.lstm_short(short_in)  # [B, T, 2H]
        h_l, _ = self.lstm_long(long_in)  # [B, T/down, H]
        
        # 上采样长程并对齐时间
        h_l_up = F.interpolate(h_l.transpose(1, 2), size=h_s.size(1), mode='linear', align_corners=False).transpose(1, 2)
        gate = self.gate(torch.cat([h_s, h_l_up], dim=-1))  # [B, T, 2H]
        h = gate * h_s + (1 - gate) * h_l_up  # 融合

        # 注意力池化（可选：对 T 做权重）
        alpha = torch.softmax(torch.mean(h, dim=-1), dim=1).unsqueeze(-1)  # [B, T, 1]
        h_clip = (h * alpha).sum(1)  # [B, 2H]
        return self.fc(h_clip)  # [B, num_classes]