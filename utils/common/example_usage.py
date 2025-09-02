"""
Example usage of common components
公共组件的使用示例
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 导入公共组件
from utils.common.label_utils import labels2cat, labels2onehot, onehot2labels, cat2labels
from utils.common.data_loaders import Dataset_CRNN, Dataset_3DCNN
from utils.common.model_components import EncoderCNN, ResCNNEncoder, DecoderRNN
from utils.common.training_utils import (
    train_epoch, validate_epoch, save_checkpoint, load_checkpoint,
    get_optimizer, get_scheduler, save_predictions, count_parameters
)

def example_crnn_model():
    """CRNN模型示例"""
    print("=== CRNN Model Example ===")
    
    # 1. 数据加载器
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 假设数据
    folders = ['video1', 'video2', 'video3']
    labels = [0, 1, 2]
    frames = list(range(16))
    
    dataset = Dataset_CRNN(
        data_path='jpegs_256',
        folders=folders,
        labels=labels,
        frames=frames,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    
    # 2. 模型组件
    encoder = EncoderCNN(
        img_x=224, img_y=224,
        fc_hidden1=512, fc_hidden2=512,
        drop_p=0.3, CNN_embed_dim=300
    )
    
    decoder = DecoderRNN(
        CNN_embed_dim=300,
        h_RNN_layers=3, h_RNN=256,
        h_FC_dim=128, drop_p=0.3,
        num_classes=101
    )
    
    # 3. 完整模型
    class CRNNModel(nn.Module):
        def __init__(self, encoder, decoder):
            super(CRNNModel, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, x):
            # x shape: (batch, time, channels, height, width)
            cnn_out = self.encoder(x)  # (batch, time, CNN_embed_dim)
            output = self.decoder(cnn_out)  # (batch, num_classes)
            return output
    
    model = CRNNModel(encoder, decoder)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 4. 训练设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name='adam', lr=0.001)
    scheduler = get_scheduler(optimizer, scheduler_name='step', step_size=20)
    
    # 5. 训练循环示例
    print("\nTraining example:")
    for epoch in range(1, 3):  # 只训练2个epoch作为示例
        # 训练
        train_loss, train_acc = train_epoch(
            model, dataloader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate_epoch(
            model, dataloader, criterion, device, epoch
        )
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%")
        print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # 保存检查点
        save_checkpoint(
            model, optimizer, epoch, val_loss,
            f'results/CRNN/outputs/checkpoint_epoch_{epoch}.pth'
        )
        
        scheduler.step()
    
    print("CRNN example completed!")

def example_resnet_crnn_model():
    """ResNet-CRNN模型示例"""
    print("\n=== ResNet-CRNN Model Example ===")
    
    # 1. 使用ResNet编码器
    encoder = ResCNNEncoder(
        fc_hidden1=512, fc_hidden2=512,
        drop_p=0.3, CNN_embed_dim=300
    )
    
    decoder = DecoderRNN(
        CNN_embed_dim=300,
        h_RNN_layers=3, h_RNN=256,
        h_FC_dim=128, drop_p=0.3,
        num_classes=101
    )
    
    class ResNetCRNNModel(nn.Module):
        def __init__(self, encoder, decoder):
            super(ResNetCRNNModel, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, x):
            cnn_out = self.encoder(x)
            output = self.decoder(cnn_out)
            return output
    
    model = ResNetCRNNModel(encoder, decoder)
    print(f"ResNet-CRNN model parameters: {count_parameters(model):,}")
    
    # 2. 标签转换示例
    print("\nLabel conversion example:")
    # 假设有标签编码器
    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # label_encoder = LabelEncoder()
    # onehot_encoder = OneHotEncoder(sparse=False)
    
    # 示例标签
    labels = ['walking', 'running', 'jumping']
    print(f"Original labels: {labels}")
    
    # 这些函数需要实际的编码器才能工作
    # cat_labels = labels2cat(label_encoder, labels)
    # onehot_labels = labels2onehot(onehot_encoder, label_encoder, labels)
    # print(f"Categorical labels: {cat_labels}")
    # print(f"One-hot labels: {onehot_labels}")
    
    print("ResNet-CRNN example completed!")

def example_3d_cnn_model():
    """3D CNN模型示例"""
    print("\n=== 3D CNN Model Example ===")
    
    # 1. 3D CNN数据加载器
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图像
    ])
    
    folders = ['video1', 'video2', 'video3']
    labels = [0, 1, 2]
    frames = list(range(16))
    
    dataset = Dataset_3DCNN(
        data_path='jpegs_256',
        folders=folders,
        labels=labels,
        frames=frames,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"3D CNN Dataset size: {len(dataset)}")
    
    # 2. 简单的3D CNN模型示例
    class Conv3DModel(nn.Module):
        def __init__(self, num_classes=101):
            super(Conv3DModel, self).__init__()
            self.conv3d = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1))
            )
            self.fc = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = self.conv3d(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = Conv3DModel()
    print(f"3D CNN model parameters: {count_parameters(model):,}")
    
    print("3D CNN example completed!")

if __name__ == "__main__":
    print("Common Components Usage Examples")
    print("="*50)
    
    # 运行示例
    example_crnn_model()
    example_resnet_crnn_model()
    example_3d_cnn_model()
    
    print("\n" + "="*50)
    print("All examples completed!")
    print("\nBenefits of using common components:")
    print("1. Code reusability - 避免重复代码")
    print("2. Consistency - 统一的接口和实现")
    print("3. Maintainability - 集中维护和更新")
    print("4. Modularity - 模块化设计，易于扩展")
