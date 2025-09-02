#!/usr/bin/env python3
"""
Model Comparison Script
比较不同模型的性能
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_model_metrics(model_name):
    """加载模型指标"""
    metrics_file = f"results/{model_name}/metrics.csv"
    if os.path.exists(metrics_file):
        return pd.read_csv(metrics_file)
    else:
        print(f"Metrics file not found: {metrics_file}")
        return None

def compare_models():
    """比较所有模型的性能"""
    models = ['Conv3D', 'CRNN', 'ResNetCRNN', 'ResNetCRNN_varylength', 'swintransformer-RNN']
    
    all_metrics = []
    for model in models:
        metrics = load_model_metrics(model)
        if metrics is not None:
            metrics['model'] = model
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("No metrics found for any model")
        return
    
    # 合并所有指标
    comparison_df = pd.concat(all_metrics, ignore_index=True)
    
    # 保存比较结果
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("模型比较结果已保存到: results/model_comparison.csv")
    
    # 打印比较结果
    print("\n" + "="*80)
    print("模型性能比较")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # 创建可视化
    create_comparison_plots(comparison_df)

def create_comparison_plots(comparison_df):
    """创建比较图表"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 准确率比较
    axes[0, 0].bar(comparison_df['model'], comparison_df['accuracy'])
    axes[0, 0].set_title('模型准确率比较')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1分数比较
    axes[0, 1].bar(comparison_df['model'], comparison_df['f1_macro'])
    axes[0, 1].set_title('模型F1分数比较 (Macro)')
    axes[0, 1].set_ylabel('F1分数')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 精确率比较
    axes[1, 0].bar(comparison_df['model'], comparison_df['precision_macro'])
    axes[1, 0].set_title('模型精确率比较 (Macro)')
    axes[1, 0].set_ylabel('精确率')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 召回率比较
    axes[1, 1].bar(comparison_df['model'], comparison_df['recall_macro'])
    axes[1, 1].set_title('模型召回率比较 (Macro)')
    axes[1, 1].set_ylabel('召回率')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("模型比较图表已保存到: results/model_comparison.png")
    
    # 创建雷达图
    create_radar_plot(comparison_df)

def create_radar_plot(comparison_df):
    """创建雷达图"""
    # 选择要比较的指标
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 为每个模型绘制雷达图
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title('模型性能雷达图', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('results/model_radar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("模型雷达图已保存到: results/model_radar_plot.png")

def main():
    parser = argparse.ArgumentParser(description='Compare Video Classification Models')
    parser.add_argument('--models', nargs='+', 
                       default=['Conv3D', 'CRNN', 'ResNetCRNN', 'ResNetCRNN_varylength', 'swintransformer-RNN'],
                       help='Models to compare')
    
    args = parser.parse_args()
    
    print("开始模型比较...")
    compare_models()
    print("模型比较完成！")

if __name__ == "__main__":
    main()
