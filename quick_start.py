#!/usr/bin/env python3
"""
Quick Start Script for Video Classification Project
快速开始脚本，帮助用户快速上手项目
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    print("检查项目依赖...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 
        'matplotlib', 'seaborn', 'scikit-learn', 'PyYAML'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("所有依赖包已安装！")
    return True

def check_data():
    """检查数据是否存在"""
    print("\n检查数据...")
    
    data_path = "jpegs_256"
    if os.path.exists(data_path):
        print(f"✓ 数据目录存在: {data_path}")
        # 检查数据文件数量
        try:
            video_count = len([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
            print(f"✓ 找到 {video_count} 个视频类别")
        except:
            print("⚠ 无法统计视频数量")
        return True
    else:
        print(f"✗ 数据目录不存在: {data_path}")
        print("请先运行数据预处理: python utils/preprocess_videos.py")
        return False

def create_directories():
    """创建必要的目录"""
    print("\n创建项目目录...")
    
    directories = [
        "results/Conv3D/outputs",
        "results/CRNN/outputs", 
        "results/ResNetCRNN/outputs",
        "results/ResNetCRNN_varylength/outputs",
        "results/swintransformer-RNN/outputs",
        "results/Conv3D/check_predictions",
        "results/CRNN/check_predictions",
        "results/ResNetCRNN/check_predictions", 
        "results/ResNetCRNN_varylength/check_predictions",
        "results/swintransformer-RNN/check_predictions"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

def run_example():
    """运行示例"""
    print("\n运行示例...")
    
    # 检查是否有预训练模型
    model_files = [
        "models/ResNetCRNN/UCF101_ResNetCRNN_fixed.py",
        "models/Conv3D/UCF101_3DCNN.py",
        "models/CRNN/UCF101_CRNN.py"
    ]
    
    available_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            model_name = model_file.split('/')[1]
            available_models.append(model_name)
    
    if not available_models:
        print("✗ 未找到可用的模型文件")
        return
    
    print(f"找到可用模型: {', '.join(available_models)}")
    
    # 选择第一个可用模型进行示例
    example_model = available_models[0]
    print(f"使用 {example_model} 作为示例模型")
    
    # 运行评估示例
    try:
        print(f"\n运行 {example_model} 评估示例...")
        result = subprocess.run([
            sys.executable, "eval.py", "--model", example_model
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 评估示例运行成功")
        else:
            print("⚠ 评估示例运行失败，可能需要先训练模型")
            print("错误信息:", result.stderr)
    except Exception as e:
        print(f"⚠ 运行示例时出错: {e}")

def show_usage():
    """显示使用说明"""
    print("\n" + "="*60)
    print("项目使用说明")
    print("="*60)
    
    print("\n1. 训练模型:")
    print("   python train.py --model ResNetCRNN --epochs 10")
    print("   python train.py --model Conv3D --epochs 10")
    
    print("\n2. 评估模型:")
    print("   python eval.py --model ResNetCRNN")
    print("   python eval.py --all  # 评估所有模型")
    
    print("\n3. 比较模型:")
    print("   python compare_models.py")
    
    print("\n4. 数据预处理:")
    print("   python utils/preprocess_videos.py --input_dir /path/to/videos")
    
    print("\n5. 查看配置:")
    print("   python utils/config_loader.py")
    
    print("\n更多信息请查看: README_REFACTORED.md")

def main():
    parser = argparse.ArgumentParser(description='Video Classification Project Quick Start')
    parser.add_argument('--skip-checks', action='store_true', help='跳过依赖和数据检查')
    parser.add_argument('--run-example', action='store_true', help='运行示例')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Video Classification Project - Quick Start")
    print("="*60)
    
    if not args.skip_checks:
        # 检查依赖
        if not check_dependencies():
            return
        
        # 检查数据
        if not check_data():
            print("\n请先准备数据，然后重新运行此脚本")
            return
    
    # 创建目录
    create_directories()
    
    # 运行示例
    if args.run_example:
        run_example()
    
    # 显示使用说明
    show_usage()
    
    print("\n" + "="*60)
    print("快速开始完成！")
    print("="*60)

if __name__ == "__main__":
    main()
