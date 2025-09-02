import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions_fixed import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import glob
from tqdm import tqdm
import time
import warnings
import argparse

# 过滤sklearn的警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 添加命令行参数
parser = argparse.ArgumentParser(description='ResNetCRNN Training with Checkpoint Support')
parser.add_argument('--resume', type=int, default=None, 
                    help='Resume training from epoch (e.g., --resume 5 to resume from epoch 5)')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Starting epoch number (default: 0)')
args = parser.parse_args()

# set path
data_path = "/data2/lpq/video-classification/jpegs_256_processed/"
action_name_path = './UCF101actions.pkl'
save_model_path = "./ResNetCRNN_ckpt/"

# 创建保存目录
os.makedirs(save_model_path, exist_ok=True)

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
epochs = 120        # training epochs
batch_size = 40  
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 28, 1


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    
    # 创建进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                leave=False, ncols=100)
    
    for batch_idx, (X, y) in enumerate(pbar):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        # 避免sklearn警告：当样本数很少时直接计算准确率
        if len(y) > 1:
            step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        else:
            step_score = 1.0 if y_pred.item() == y.item() else 0.0
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # 更新进度条
        avg_loss = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
        avg_acc = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{avg_acc:.2%}',
            'Batch': f'{batch_idx+1}/{len(train_loader)}'
        })

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    
    # 创建进度条
    pbar = tqdm(test_loader, desc='[Validation]', leave=False, ncols=100)
    
    with torch.no_grad():
        for X, y in pbar:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    # 避免sklearn警告
    if len(all_y) > 1:
        test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    else:
        test_score = 1.0 if all_y_pred.item() == all_y.item() else 0.0

    # show information
    print(f'\nTest set ({len(all_y)} samples): Average loss: {test_loss:.4f}, Accuracy: {test_score:.2%}')

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))
    print(f"Epoch {epoch + 1} model saved!")

    return test_loss, test_score


def load_checkpoint(cnn_encoder, rnn_decoder, optimizer, epoch, save_model_path, device):
    """
    加载指定epoch的checkpoint
    """
    try:
        # 构建checkpoint文件路径
        cnn_path = os.path.join(save_model_path, f'cnn_encoder_epoch{epoch}.pth')
        rnn_path = os.path.join(save_model_path, f'rnn_decoder_epoch{epoch}.pth')
        optimizer_path = os.path.join(save_model_path, f'optimizer_epoch{epoch}.pth')
        
        # 检查文件是否存在
        if not all(os.path.exists(path) for path in [cnn_path, rnn_path, optimizer_path]):
            print(f"Warning: Checkpoint files for epoch {epoch} not found!")
            return False
        
        # 加载状态字典
        cnn_state_dict = torch.load(cnn_path, map_location=device)
        rnn_state_dict = torch.load(rnn_path, map_location=device)
        
        # 检查是否是DataParallel格式（包含module前缀）
        is_dataparallel = list(cnn_state_dict.keys())[0].startswith('module.')
        current_is_dataparallel = torch.cuda.device_count() > 1
        
        # 根据checkpoint格式和当前环境调整模型
        if is_dataparallel and not current_is_dataparallel:
            # checkpoint是多GPU格式，但当前是单GPU环境
            print("Converting DataParallel checkpoint to single GPU format...")
            # 移除module前缀
            cnn_state_dict = {k.replace('module.', ''): v for k, v in cnn_state_dict.items()}
            rnn_state_dict = {k.replace('module.', ''): v for k, v in rnn_state_dict.items()}
        elif not is_dataparallel and current_is_dataparallel:
            # checkpoint是单GPU格式，但当前是多GPU环境
            print("Converting single GPU checkpoint to DataParallel format...")
            # 添加module前缀
            cnn_state_dict = {f'module.{k}': v for k, v in cnn_state_dict.items()}
            rnn_state_dict = {f'module.{k}': v for k, v in rnn_state_dict.items()}
        
        # 加载模型权重
        cnn_encoder.load_state_dict(cnn_state_dict)
        rnn_decoder.load_state_dict(rnn_state_dict)
        
        # 加载优化器状态
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
        
        print(f"Successfully loaded checkpoint from epoch {epoch}")
        return True
        
    except Exception as e:
        print(f"Error loading checkpoint from epoch {epoch}: {str(e)}")
        return False


def load_training_history():
    """
    加载训练历史记录
    """
    try:
        epoch_train_losses = np.load('./CRNN_epoch_training_losses.npy').tolist()
        epoch_train_scores = np.load('./CRNN_epoch_training_scores.npy').tolist()
        epoch_test_losses = np.load('./CRNN_epoch_test_loss.npy').tolist()
        epoch_test_scores = np.load('./CRNN_epoch_test_score.npy').tolist()
        print("Successfully loaded training history")
        return epoch_train_losses, epoch_train_scores, epoch_test_losses, epoch_test_scores
    except FileNotFoundError:
        print("Training history files not found, starting fresh")
        return [], [], [], []


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# load UCF101 actions names
with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
print("Total classes:", len(le.classes_))
print("Classes:", list(le.classes_))

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# 根据实际数据结构收集所有视频文件
print("Collecting video files...")
all_video_folders = []
all_video_labels = []

# 遍历每个动作类别
for action_name in action_names:
    action_path = os.path.join(data_path, action_name)
    if os.path.exists(action_path):
        # 获取该动作下的所有视频文件夹
        video_folders = glob.glob(os.path.join(action_path, "v_*"))
        for video_folder in video_folders:
            # 检查是否有足够的帧
            frame_files = glob.glob(os.path.join(video_folder, "frame*.jpg"))
            if len(frame_files) >= end_frame:
                all_video_folders.append(video_folder)
                all_video_labels.append(action_name)

print(f"Total videos found: {len(all_video_folders)}")

# 转换为类别标签
all_y_list = labels2cat(le, all_video_labels)

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_video_folders, all_y_list, test_size=0.25, random_state=42)

print(f"Training samples: {len(train_list)}")
print(f"Testing samples: {len(test_list)}")

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

train_set, valid_set = Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform), \
                       Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)

# 处理checkpoint恢复
start_epoch = args.start_epoch
if args.resume is not None:
    print(f"\nAttempting to resume training from epoch {args.resume}...")
    if load_checkpoint(cnn_encoder, rnn_decoder, optimizer, args.resume, save_model_path, device):
        start_epoch = args.resume
        print(f"Training will resume from epoch {start_epoch + 1}")
    else:
        print("Failed to load checkpoint, starting from scratch")
        start_epoch = 0

# 加载训练历史记录
epoch_train_losses, epoch_train_scores, epoch_test_losses, epoch_test_scores = load_training_history()

# 如果从checkpoint恢复，需要截断历史记录到正确的长度
if start_epoch > 0:
    epoch_train_losses = epoch_train_losses[:start_epoch]
    epoch_train_scores = epoch_train_scores[:start_epoch]
    epoch_test_losses = epoch_test_losses[:start_epoch]
    epoch_test_scores = epoch_test_scores[:start_epoch]

# start training
print(f"\n{'='*60}")
print(f"Starting Training - ResNetCRNN on UCF101")
print(f"Starting from epoch: {start_epoch + 1}")
print(f"Total epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Device: {device}")
print(f"{'='*60}\n")

start_time = time.time()

for epoch in range(start_epoch, epochs):
    epoch_start_time = time.time()
    
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # 计算平均指标
    avg_train_loss = np.mean(train_losses)
    avg_train_acc = np.mean(train_scores)
    epoch_time = time.time() - epoch_start_time
    
    # 显示epoch总结
    print(f"Epoch {epoch+1:3d}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Train Acc: {avg_train_acc:.2%} | "
          f"Test Loss: {epoch_test_loss:.4f} | "
          f"Test Acc: {epoch_test_score:.2%} | "
          f"Time: {epoch_time:.1f}s")

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Training completed in {total_time/3600:.1f} hours")
print(f"Best test accuracy: {max(epoch_test_scores):.2%}")
print(f"{'='*60}")

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
# 计算正确的epoch范围
epoch_range = np.arange(start_epoch + 1, start_epoch + len(epoch_train_losses) + 1)
plt.plot(epoch_range, A[:, -1])  # train loss (on epoch end)
plt.plot(epoch_range, C)         #  test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure
plt.subplot(122)
plt.plot(epoch_range, B[:, -1])  # train accuracy (on epoch end)
plt.plot(epoch_range, D)         #  test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = "./fig_UCF101_ResNetCRNN.png"
plt.savefig(title, dpi=600)
plt.show() 