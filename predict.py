#!/usr/bin/env python3
"""
Single-sample prediction script for video classification
使用配置与权重，对单个样本帧目录（.../Action/v_*）进行Top-K预测
"""

import argparse
import os
import sys
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
from common.data_loaders import Dataset_CRNN, Dataset_3DCNN, Dataset_SwinCRNN
from common.model_components import EncoderCNN, ResCNNEncoder, DecoderRNN, SwinTransformerEncoder


def main():
    parser = argparse.ArgumentParser(description='Single-sample prediction')
    parser.add_argument('--model', type=str, required=True,
                        choices=['Conv3D', 'CRNN', 'ResNetCRNN', 'swintransformer-RNN'],
                        help='Model type to run')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config for data paths')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint')
    parser.add_argument('--sample', type=str, required=True, help='Path to a video frames folder (v_* dir)')
    parser.add_argument('--topk', type=int, default=5, help='Top-K predictions to display')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get('data', {}) if isinstance(cfg, dict) else {}
    data_path = data_cfg.get('data_path')
    action_name_path = data_cfg.get('action_name_path', 'data/UCF101actions.pkl')
    if not data_path:
        raise ValueError('data.data_path missing in config')

    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build input tensor
    if args.model in ['CRNN', 'ResNetCRNN']:
        resize_hw = 224
        transform = T.Compose([T.Resize([resize_hw, resize_hw]), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        frames = list(range(1, 29, 1))
        ds = Dataset_CRNN(data_path=data_path, folders=[os.path.relpath(args.sample, data_path)], labels=[0], frames=frames, transform=transform)
        x, _ = ds[0]
        x = x.unsqueeze(0).to(device)
        if args.model == 'ResNetCRNN':
            encoder = ResCNNEncoder(fc_hidden1=1024, fc_hidden2=768, drop_p=0.0, CNN_embed_dim=512)
        else:
            encoder = EncoderCNN(img_x=224, img_y=224, fc_hidden1=1024, fc_hidden2=768, drop_p=0.0, CNN_embed_dim=512)
        decoder = DecoderRNN(CNN_embed_dim=512, h_RNN_layers=3, h_RNN=512, h_FC_dim=256, drop_p=0.0, num_classes=len(action_names))
        model = nn.Sequential(encoder, decoder)
    elif args.model == 'swintransformer-RNN':
        resize_hw = 224
        transform = T.Compose([T.Resize([resize_hw, resize_hw]), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        frames = list(range(1, 29, 1))
        ds = Dataset_SwinCRNN(data_path=data_path, folders=[os.path.relpath(args.sample, data_path)], labels=[0], frames=frames, transform=transform)
        x, _ = ds[0]
        x = x.unsqueeze(0).to(device)
        encoder = SwinTransformerEncoder(fc_hidden1=1024, fc_hidden2=768, drop_p=0.0, CNN_embed_dim=512, 
                                       swin_model_name='swin_tiny_patch4_window7_224')
        decoder = DecoderRNN(CNN_embed_dim=512, h_RNN_layers=3, h_RNN=512, h_FC_dim=256, drop_p=0.0, num_classes=len(action_names))
        model = nn.Sequential(encoder, decoder)
    elif args.model == 'Conv3D':
        resize_hw = 112
        transform = T.Compose([T.Resize([resize_hw, resize_hw]), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
        frames = list(range(1, 29, 1))
        ds = Dataset_3DCNN(data_path=data_path, folders=[os.path.relpath(args.sample, data_path)], labels=[0], frames=frames, transform=transform)
        x, _ = ds[0]
        x = x.unsqueeze(0).to(device)
        conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32,64, kernel_size=(3,3,3), padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(64,128, kernel_size=(3,3,3), padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(128,256, kernel_size=(3,3,3), padding=1), nn.BatchNorm3d(256), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, len(action_names)))
        class Conv3DWrap(nn.Module):
            def __init__(self, c, f):
                super().__init__(); self.c=c; self.f=f
            def forward(self, x):
                x=self.c(x); x=x.view(x.size(0), -1); return self.f(x)
        model = Conv3DWrap(conv3d, fc)
    else:
        raise NotImplementedError

    model = model.to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # 支持新的统一checkpoint格式
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            # 重构后的格式
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif 'cnn_encoder_state_dict' in ckpt and 'rnn_decoder_state_dict' in ckpt:
            # ResNetCRNN 分离格式
            if args.model == 'ResNetCRNN':
                encoder.load_state_dict(ckpt['cnn_encoder_state_dict'])
                decoder.load_state_dict(ckpt['rnn_decoder_state_dict'])
            else:
                print('Warning: checkpoint format not compatible with this model')
                return
        elif 'swin_encoder_state_dict' in ckpt and 'rnn_decoder_state_dict' in ckpt:
            # SwinTransformer-RNN 分离格式
            if args.model == 'swintransformer-RNN':
                encoder.load_state_dict(ckpt['swin_encoder_state_dict'])
                decoder.load_state_dict(ckpt['rnn_decoder_state_dict'])
            else:
                print('Warning: checkpoint format not compatible with this model')
                return
        else:
            print('Warning: unknown checkpoint format')
            return
    else:
        # 旧格式，直接加载
        try:
            model.load_state_dict(ckpt, strict=False)
        except Exception:
            print('Warning: could not load checkpoint strictly; proceeding if layers match')

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    topk = min(args.topk, len(action_names))
    idx = np.argsort(-probs)[:topk]
    print('Top-{} predictions:'.format(topk))
    for i in idx:
        print(f"{action_names[i]}: {probs[i]:.4f}")

if __name__ == '__main__':
    main()


