import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# set visible CUDA device
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Configuration class
class Config:
    def __init__(self):
        # Default paths
        self.data_path = "/data2/lpq/video-classification/jpegs_256_processed/"
        self.action_name_path = 'data/UCF101actions.pkl'
        self.frame_slice_file = 'data/UCF101_frame_count.pkl'
        self.save_model_path = "results/ResNetCRNN_varylength/outputs"
        
        # EncoderCNN architecture
        self.CNN_fc_hidden1, self.CNN_fc_hidden2 = 1024, 768
        self.CNN_embed_dim = 512   # latent dim extracted by 2D CNN
        self.res_size = 224        # ResNet image size
        self.dropout_p = 0.0       # dropout probability
        
        # DecoderRNN architecture
        self.RNN_hidden_layers = 3
        self.RNN_hidden_nodes = 512
        self.RNN_FC_dim = 256
        
        # training parameters
        self.k = 101             # number of target category
        self.epochs = 150        # training epochs
        self.batch_size = 120
        self.learning_rate = 1e-3
        self.lr_patience = 15
        self.log_interval = 10   # interval for displaying training info
        
        # Select frames to begin & end in videos
        self.select_frame = {'begin': 1, 'end': 100, 'skip': 2}

# Global config object
config = Config()

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    epoch_loss, all_y, all_y_pred = 0, [], []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, X_lengths, y) in enumerate(train_loader):
        # distribute data to device
        X, X_lengths, y = X.to(device), X_lengths.to(device).view(-1, ), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X), X_lengths)   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)  # mini-batch loss
        epoch_loss += F.cross_entropy(output, y, reduction='sum').item()  # sum up mini-batch loss

        y_pred = torch.max(output, 1)[1]  # y_pred != output

        # collect all y and y_pred in all mini-batches
        all_y.extend(y)
        all_y_pred.extend(y_pred)

        # to compute accuracy
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % config.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    epoch_loss /= len(train_loader)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    epoch_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    return epoch_loss, epoch_score


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y, all_y_pred = [], []
    with torch.no_grad():
        for X, X_lengths, y in test_loader:
            # distribute data to device
            X, X_lengths, y = X.to(device), X_lengths.to(device).view(-1, ), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X), X_lengths)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up minibatch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
            check_mkdir(config.save_model_path)
        
        # 统一保存为单个checkpoint文件
        checkpoint = {
            'epoch': epoch + 1,
            'cnn_encoder_state_dict': cnn_encoder.state_dict(),
            'rnn_decoder_state_dict': rnn_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_train_loss,
            'score': epoch_train_score
        }
        torch.save(checkpoint, os.path.join(config.save_model_path, f'checkpoint_epoch_{epoch + 1}.pth'))
        
        # 同时保存最佳模型
        if epoch_train_score > getattr(config, 'best_score', 0):
            config.best_score = epoch_train_score
            torch.save(checkpoint, os.path.join(config.save_model_path, 'best_model.pth'))
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
    params = {'batch_size': config.batch_size, 'shuffle': True, 'num_workers': 8, 'pin_memory': True} if use_cuda else {}


# load UCF101 actions names
with open(config.action_name_path, 'rb') as f:
    action_names = pickle.load(f)

# load UCF101 video length
with open(config.frame_slice_file, 'rb') as f:
    slice_count = pickle.load(f)

# convert labels -> category
le = LabelEncoder()
le.fit(action_names)

# show how many classes there are
list(le.classes_)

# convert category -> 1-hot
action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

actions = []
fnames = os.listdir(config.data_path)

all_names = []
all_length = []    # each video length
for f in fnames:
    loc1 = f.find('v_')
    loc2 = f.find('_g')
    actions.append(f[(loc1 + 2): loc2])

    all_names.append(os.path.join(config.data_path, f))
    all_length.append(slice_count[f])

# list all data files
all_X_list = list(zip(all_names, all_length))   # video (names, length)
all_y_list = labels2cat(le, actions)            # video labels

# all_X_list = all_X_list[:200]   # use only a few samples for testing
# all_y_list = all_y_list[:200]

# train, test split
train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

transform = transforms.Compose([transforms.Resize([config.res_size, config.res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_set, valid_set = Dataset_CRNN_varlen(config.data_path, train_list, train_label, config.select_frame, transform=transform), \
                       Dataset_CRNN_varlen(config.data_path, test_list, test_label, config.select_frame, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=config.CNN_fc_hidden1, fc_hidden2=config.CNN_fc_hidden2, drop_p=config.dropout_p, CNN_embed_dim=config.CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN_varlen(CNN_embed_dim=config.CNN_embed_dim, h_RNN_layers=config.RNN_hidden_layers, h_RNN=config.RNN_hidden_nodes,
                         h_FC_dim=config.RNN_FC_dim, drop_p=config.dropout_p, num_classes=config.k).to(device)

# Combine all EncoderCNN + DecoderRNN parameters
print("Using", torch.cuda.device_count(), "GPU!")
if torch.cuda.device_count() > 1:
    # Parallelize model to multiple GPUs
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=config.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.lr_patience, min_lr=1e-10, verbose=True)

# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(config.epochs):
    # train, test model
    epoch_train_loss, epoch_train_score = train(config.log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)
    scheduler.step(epoch_test_loss)

    # save results
    epoch_train_losses.append(epoch_train_loss)
    epoch_train_scores.append(epoch_train_score)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save(os.path.join(config.save_model_path, 'CRNN_varlen_epoch_training_loss.npy'), A)
    np.save(os.path.join(config.save_model_path, 'CRNN_varlen_epoch_training_score.npy'), B)
    np.save(os.path.join(config.save_model_path, 'CRNN_varlen_epoch_test_loss.npy'), C)
    np.save(os.path.join(config.save_model_path, 'CRNN_varlen_epoch_test_score.npy'), D)

# plot
fig = plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(np.arange(1, config.epochs + 1), A)  # train loss (on epoch end)
plt.plot(np.arange(1, config.epochs + 1), C)  # test loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc="upper left")
# 2nd figure (accuracy)
plt.subplot(122)
plt.plot(np.arange(1, config.epochs + 1), B)  # train accuracy (on epoch end)
plt.plot(np.arange(1, config.epochs + 1), D)  # test accuracy (on epoch end)
plt.title("training scores")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc="upper left")
title = os.path.join(config.save_model_path, "fig_UCF101_ResNetCRNN.png")
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()

# 如果提供了配置文件，则更新配置
if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='ResNetCRNN_varylength Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 更新配置
        if 'data' in config_dict:
            data = config_dict['data']
            if 'data_path' in data:
                config.data_path = data['data_path']
            if 'action_name_path' in data:
                config.action_name_path = data['action_name_path']
        
        if 'checkpoint' in config_dict:
            checkpoint = config_dict['checkpoint']
            if 'save_dir' in checkpoint:
                config.save_model_path = checkpoint['save_dir']
        
        if 'training' in config_dict:
            training = config_dict['training']
            if 'epochs' in training:
                config.epochs = training['epochs']
            if 'batch_size' in training:
                config.batch_size = training['batch_size']
            if 'learning_rate' in training:
                config.learning_rate = training['learning_rate']
    
    print("Configuration:")
    print(f"  Data path: {config.data_path}")
    print(f"  Action name path: {config.action_name_path}")
    print(f"  Save model path: {config.save_model_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # 创建保存目录
    os.makedirs(config.save_model_path, exist_ok=True)
