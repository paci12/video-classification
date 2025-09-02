#!/usr/bin/env python3
"""
Video Classification Evaluation Script
支持多种模型的评估：Conv3D, CRNN, ResNetCRNN, ResNetCRNN_varylength, SwinTransformer-RNN
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

# utils path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)
from common.data_loaders import Dataset_CRNN, Dataset_3DCNN, Dataset_SwinCRNN
from common.model_components import EncoderCNN, ResCNNEncoder, DecoderRNN, SwinTransformerEncoder

def generate_predictions(model_name, epoch=None, save_predictions=False):
    """
    通过指定权重文件生成测试集的预测结果
    
    Args:
        model_name: 模型名称
        epoch: 指定的epoch，如果为None则使用最佳模型
        save_predictions: 是否保存预测结果，默认False
    
    Returns:
        预测结果DataFrame或None
    """
    print(f"正在为 {model_name} 生成预测结果...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 从配置文件读取checkpoint路径
    config_path = f"configs/{model_name}_train.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        ckpt_dir = config.get('checkpoint', {}).get('save_dir', f"results/{model_name}/result/ckpt")
        print(f"从配置文件读取checkpoint路径: {ckpt_dir}")
    else:
        # 如果配置文件不存在，使用默认路径
        ckpt_dir = f"results/{model_name}/result/ckpt"
        print(f"配置文件不存在，使用默认路径: {ckpt_dir}")
    
    if epoch is not None:
        checkpoint_path = f"{ckpt_dir}/checkpoint_epoch_{epoch}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint文件不存在: {checkpoint_path}")
            return None
    else:
        checkpoint_path = f"{ckpt_dir}/best_model.pth"
        if not os.path.exists(checkpoint_path):
            print(f"最佳模型文件不存在: {checkpoint_path}")
            return None
    
    # 加载动作名称
    action_names_file = f"models/{model_name}/UCF101actions.pkl"
    if os.path.exists(action_names_file):
        with open(action_names_file, 'rb') as f:
            action_names = pickle.load(f)
        print(f"加载动作名称: {len(action_names)} 个类别")
    else:
        print("动作名称文件不存在，使用数字标签")
        action_names = [f"Class_{i}" for i in range(101)]
    
    # 根据模型类型创建模型和数据加载器
    if model_name == 'swintransformer-RNN':
        # 创建SwinTransformer-RNN模型
        swin_encoder = SwinTransformerEncoder(
            fc_hidden1=1024, fc_hidden2=768, drop_p=0.0, CNN_embed_dim=512,
            swin_model_name='swin_tiny_patch4_window7_224'
        )
        rnn_decoder = DecoderRNN(
            CNN_embed_dim=512, h_RNN_layers=3, h_RNN=512, h_FC_dim=256,
            drop_p=0.0, num_classes=len(action_names)
        )
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'swin_encoder_state_dict' in checkpoint and 'rnn_decoder_state_dict' in checkpoint:
            swin_encoder.load_state_dict(checkpoint['swin_encoder_state_dict'])
            rnn_decoder.load_state_dict(checkpoint['rnn_decoder_state_dict'])
        else:
            print("Checkpoint格式不正确")
            return None
        
        swin_encoder = swin_encoder.to(device)
        rnn_decoder = rnn_decoder.to(device)
        
        # 创建数据加载器（这里需要根据实际数据路径调整）
        data_path = "/data2/lpq/video-classification/jpegs_256_processed/"
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 这里需要根据实际情况创建测试数据集
        # 暂时返回None，表示需要手动实现数据加载部分
        print("注意：数据加载部分需要根据实际情况实现")
        return None
        
    elif model_name == 'ResNetCRNN':
        # 创建ResNetCRNN模型
        encoder = ResCNNEncoder(fc_hidden1=1024, fc_hidden2=768, drop_p=0.0, CNN_embed_dim=512)
        decoder = DecoderRNN(CNN_embed_dim=512, h_RNN_layers=3, h_RNN=512, h_FC_dim=256, drop_p=0.0, num_classes=len(action_names))
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'cnn_encoder_state_dict' in checkpoint and 'rnn_decoder_state_dict' in checkpoint:
            encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
            decoder.load_state_dict(checkpoint['rnn_decoder_state_dict'])
        else:
            print("Checkpoint格式不正确")
            return None
        
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        # 这里需要根据实际情况创建测试数据集
        print("注意：数据加载部分需要根据实际情况实现")
        return None
        
    else:
        print(f"暂不支持模型类型: {model_name}")
        return None
    
    # 如果save_predictions为True，保存预测结果
    if save_predictions:
        # 这里需要实现实际的预测和保存逻辑
        print("预测结果保存功能需要进一步实现")
    
    return None

def load_predictions(model_name, epoch=None, save_predictions=False):
    """加载指定模型的预测结果"""
    if epoch is not None:
        # 如果指定了epoch，尝试加载对应epoch的预测结果
        pred_file = f"results/{model_name}/result/check_predictions/{model_name}_epoch_{epoch}_epoch_latest_videos_prediction.pkl"
        if os.path.exists(pred_file):
            with open(pred_file, 'rb') as f:
                predictions_df = pickle.load(f)
            print(f"加载 {model_name} epoch {epoch} 预测结果: {len(pred_file)} 个样本")
            return predictions_df
        else:
            print(f"Epoch {epoch} 的预测结果文件不存在: {pred_file}")
    
    # 如果没有指定epoch或指定epoch的文件不存在，尝试加载默认的预测结果
    pred_file = f"results/{model_name}/result/check_predictions/{model_name}_epoch_latest_videos_prediction.pkl"
    if os.path.exists(pred_file):
        with open(pred_file, 'rb') as f:
            predictions_df = pickle.load(f)
        print(f"加载 {model_name} 默认预测结果: {len(predictions_df)} 个样本")
        return predictions_df
    else:
        print(f"预测结果文件不存在: {pred_file}")
        print("尝试自动生成预测结果...")
        # 自动生成预测结果
        return generate_predictions(model_name, epoch, save_predictions=save_predictions)

def calculate_model_metrics(model_name, epoch=None, save_predictions=False):
    """计算指定模型的评估指标"""
    predictions_df = load_predictions(model_name, epoch, save_predictions)
    if predictions_df is None:
        return None
    
    # 加载动作名称
    action_names_file = f"configs/data/UCF101actions.pkl"
    if os.path.exists(action_names_file):
        with open(action_names_file, 'rb') as f:
            action_names = pickle.load(f)
        print(f"加载动作名称: {len(action_names)} 个类别")
    else:
        print("动作名称文件不存在，使用数字标签")
        action_names = [f"Class_{i}" for i in range(101)]
    
    # 检查数据列
    print(f"预测结果列名: {predictions_df.columns.tolist()}")
    
    if 'y' in predictions_df.columns and 'y_pred' in predictions_df.columns:
        y_true = predictions_df['y'].values
        y_pred = predictions_df['y_pred'].values
        
        # 导入评估函数
        sys.path.append('utils')
        from calculate_metrics import calculate_metrics, plot_confusion_matrix, plot_class_performance
        
        # 计算指标
        print(f"\n计算 {model_name} 评估指标...")
        metrics = calculate_metrics(y_true, y_pred)
        
        # 打印结果
        print(f"\n{model_name} 模型评估结果")
        print("="*60)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"精确率 (Precision) - Macro: {metrics['precision_macro']:.4f}")
        print(f"召回率 (Recall) - Macro: {metrics['recall_macro']:.4f}")
        print(f"F1分数 - Macro: {metrics['f1_macro']:.4f}")
        
        # 生成可视化
        if epoch is not None:
            # 指定epoch时，在outputs目录下创建eval_epoch_{epoch}子目录
            output_dir = f"results/{model_name}/result/outputs/eval_epoch_{epoch}"
        else:
            # 不指定epoch时，在outputs目录下创建eval_latest子目录
            output_dir = f"results/{model_name}/result/outputs/eval_latest"
        
        # 创建目录（如果不存在），不会删除原有内容
        os.makedirs(output_dir, exist_ok=True)
        
        plot_confusion_matrix(y_true, y_pred, action_names, 
                            save_path=f"{output_dir}/confusion_matrix.png")
        plot_class_performance(y_true, y_pred, action_names, 
                             save_path=f"{output_dir}/class_performance.png")
        
        return metrics
    else:
        print("预测结果文件格式不正确")
        return None

def main():
    parser = argparse.ArgumentParser(description='Video Classification Evaluation')
    parser.add_argument('--model', type=str, required=True,
                       choices=['Conv3D', 'CRNN', 'ResNetCRNN', 'ResNetCRNN_varylength', 'swintransformer-RNN'],
                       help='Model type to evaluate')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Specific epoch to evaluate (optional)')
    parser.add_argument('--save-predictions', action='store_true', default=False,
                       help='Save generated predictions to check_predictions directory')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all models using their latest/best models (ignores --epoch)')
    # 单样本预测现已迁移到 predict.py
    
    args = parser.parse_args()
    
    if args.all:
        models = ['Conv3D', 'CRNN', 'ResNetCRNN', 'ResNetCRNN_varylength', 'swintransformer-RNN']
        all_metrics = {}
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"评估模型: {model}")
            print(f"{'='*60}")
            # --all 模式默认使用最新/最佳模型（不指定epoch）
            metrics = calculate_model_metrics(model, None, args.save_predictions)
            if metrics:
                all_metrics[model] = metrics
        
        # 保存所有模型的比较结果
        if all_metrics:
            comparison_df = pd.DataFrame(all_metrics).T
            comparison_df.to_csv('results/model_comparison.csv')
            print(f"\n模型比较结果已保存到: results/model_comparison.csv")
            print(comparison_df)
    else:
        metrics = calculate_model_metrics(args.model, args.epoch, args.save_predictions)
        if metrics:
            # 保存单个模型的结果
            if args.epoch is not None:
                # 指定epoch时，在outputs目录下创建eval_epoch_{epoch}子目录
                output_dir = f"results/{args.model}/result/outputs/eval_epoch_{args.epoch}"
            else:
                # 不指定epoch时，在outputs目录下创建eval_latest子目录
                output_dir = f"results/{args.model}/result/outputs/eval_latest"
            
            # 创建目录（如果不存在），不会删除原有内容
            os.makedirs(output_dir, exist_ok=True)
            
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)
            print(f"评估结果已保存到: {output_dir}/metrics.csv")

if __name__ == "__main__":
    main()
