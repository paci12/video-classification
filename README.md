# Video Classification (Refactored)

## 概览
- 统一目录与模块化：`models/`, `utils/common`, `configs/`, `results/`
- 公共组件复用：数据加载、模型组件、训练/验证/评估工具集中在 `utils/common`
- 统一脚本：训练 `train.py`、评估 `eval.py`、单样本预测 `predict.py`
- 配置驱动：所有路径与超参均来自 `configs/*.yaml`
- 自动目录管理：训练时自动备份原有结果并创建新目录
- **Resume训练支持**：支持从指定epoch继续训练，智能目录管理

## 目录结构
```
video-classification/
├── models/
│   ├── Conv3D/
│   ├── CRNN/
│   ├── ResNetCRNN/
│   ├── ResNetCRNN_varylength/
│   └── swintransformer_rnn/           # 注意：下划线命名
├── utils/
│   ├── common/                         # 标签/数据/模型/训练 公共组件
│   ├── config_loader.py               # 配置加载器
│   ├── calculate_metrics.py           # 评估指标计算
│   ├── preprocess_videos.py          # 视频预处理
│   └── test_data_format.py           # 数据格式验证
├── configs/
│   ├── data/
│   │   └── UCF101actions.pkl         # 动作类别文件
│   ├── Conv3D_train.yaml
│   ├── CRNN_train.yaml
│   ├── ResNetCRNN_train.yaml
│   ├── ResNetCRNN_varylength_train.yaml
│   └── swintransformer-RNN_train.yaml
├── results/                           # 训练结果根目录
├── train.py                           # 统一训练脚本
├── eval.py                            # 模型评估脚本
├── predict.py                         # 单样本预测脚本
└── README.md                          # 本文件
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 数据预处理
-预处理脚本位于utils/run_preprocessing.sh
-用于抽取视频帧将视频文件转化成一个包含视频帧的目录

## 配置说明（YAML）

### 配置文件位置和命名
- 所有配置文件位于 `configs/` 目录
- 命名格式：`{ModelName}_train.yaml`
- 配置文件与模型名称对应（注意大小写和连字符）

### 配置文件结构
所有配置文件都遵循相同的结构，包含以下主要部分：

#### 1. model 配置
```yaml
model:
  name: "模型名称"
  # 模型特定参数
  cnn_encoder: "resnet50"        # 编码器类型
  rnn_decoder: "lstm"            # RNN类型
  hidden_size: 512               # 隐藏层大小
  num_layers: 2                  # 层数
  dropout: 0.5                   # Dropout率
```

#### 2. training 配置
```yaml
training:
  epochs: 50                     # 训练轮数
  batch_size: 32                 # 批次大小
  learning_rate: 0.001           # 学习率
  weight_decay: 1e-4             # 权重衰减
  optimizer: "adam"              # 优化器
  scheduler: "step"              # 学习率调度器
  step_size: 20                  # 调度器步长
  gamma: 0.1                     # 学习率衰减因子
  # 可选配置（已注释，可开启）
  # amp: true                    # 混合精度训练
  # grad_accum_steps: 1          # 梯度累积步数
  # seed: 42                     # 随机种子
  # early_stopping:
  #   enabled: true              # 启用早停
  #   patience: 10               # 忍耐轮数
  #   min_delta: 0.0             # 最小改进幅度
  # monitor: "val_loss"          # 监控指标
```

#### 3. data 配置
```yaml
data:
  dataset: "UCF101"              # 数据集名称
  data_path: "/path/to/processed/frames/"  # 预处理帧根目录
  action_name_path: "configs/data/UCF101actions.pkl"  # 类别文件路径
  num_frames: 16                 # 每个视频的帧数
  frame_size: 224                # 帧尺寸
  num_classes: 101               # 类别数量
  train_split: 0.8               # 训练集比例
  val_split: 0.2                 # 验证集比例
  # 可选配置
  # num_workers: 4               # DataLoader worker数量
  # pin_memory: true             # CUDA下建议开启
  # persistent_workers: true     # 长生命周期worker
```

#### 4. augmentation 配置
```yaml
augmentation:
  horizontal_flip: true          # 水平翻转
  random_crop: true              # 随机裁剪
  color_jitter: true             # 颜色抖动
  normalize:                      # 标准化
    mean: [0.485, 0.456, 0.406] # RGB均值
    std: [0.229, 0.224, 0.225]  # RGB标准差
  # 高级增强（部分模型支持）
  # mixup: true                  # Mixup增强
  # cutmix: true                 # CutMix增强
```

#### 5. checkpoint 配置
```yaml
checkpoint:
  save_dir: "results/ModelName/result"  # 保存根目录
  save_freq: 5                   # 保存频率（每N个epoch）
  pretrained: true               # 是否使用预训练权重
  # 注意：resume参数通过命令行指定，不在配置文件中
  # 可选配置
  # best_metric: "val_loss"      # 最佳模型指标
  # save_predictions: true       # 是否保存预测结果
```

#### 6. logging 配置
```yaml
logging:
  log_dir: "results/ModelName/result/outputs/logs"  # 日志目录
  tensorboard: true              # 是否启用TensorBoard
  wandb: false                   # 是否启用Weights & Biases
```

### 各模型配置文件特点

#### Conv3D 配置
- 使用3D卷积网络
- 适合时序建模
- 内存占用较高

#### CRNN 配置
- CNN + RNN 架构
- 使用VGG16作为编码器
- 平衡性能和效率

#### ResNetCRNN 配置
- ResNet + RNN 架构
- 使用ResNet50作为编码器
- 性能优秀，训练稳定

#### SwinTransformer-RNN 配置
- SwinTransformer + RNN 架构
- 使用Vision Transformer
- 需要较小的batch size
- 使用AdamW优化器和余弦退火调度器

### 配置文件示例对比

#### 基础配置（CRNN）
```yaml
checkpoint:
  save_dir: "results/CRNN/result"
logging:
  log_dir: "results/CRNN/result/outputs/logs"
```

#### 高级配置（SwinTransformer-RNN）
```yaml
checkpoint:
  save_dir: "results/swintransformer-RNN/result"
logging:
  log_dir: "results/swintransformer-RNN/result/outputs/logs"
training:
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 5
```

## 训练

### 基本训练命令
```bash
# 使用配置文件训练
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml
python train.py --model CRNN       --config configs/CRNN_train.yaml
python train.py --model Conv3D     --config configs/Conv3D_train.yaml
python train.py --model swintransformer-RNN --config configs/swintransformer-RNN_train.yaml

# 可选参数覆盖（会覆盖配置文件中的设置）
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --batch_size 8
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --epochs 100
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --lr 0.0001
```

### Resume训练（继续训练）
```bash
# 从指定epoch继续训练（推荐）
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --resume 45

# 从epoch 30继续训练
python train.py --model CRNN --config configs/CRNN_train.yaml --resume 30

# 从epoch 20继续训练SwinTransformer
python train.py --model swintransformer-RNN --config configs/swintransformer-RNN_train.yaml --resume 20
```

### 目录管理参数
```bash
# 强制不备份目录（直接覆盖）
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --no_backup

# 调试时避免创建备份
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --no_backup

# Resume训练时自动跳过备份
python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml --resume 45
```

### 配置文件优先级
1. **命令行参数**：最高优先级，会覆盖配置文件
2. **配置文件**：基础配置，包含所有必要参数
3. **默认值**：如果配置文件中没有指定，使用代码默认值

### 训练特性
- **自动目录管理**：训练开始前自动备份原有 `result` 目录为 `result_{创建时间}`，然后创建新的 `result` 目录
- **智能Resume支持**：使用 `--resume` 时自动跳过目录备份，直接使用现有目录和checkpoint
- **训练历史保存**：自动保存每个epoch的训练损失、准确率等指标
- **训练曲线图**：自动生成训练曲线图并保存
- **预测结果保存**：训练完成后保存验证集预测结果

### Resume训练工作原理
1. **检查checkpoint文件**：验证指定epoch的checkpoint是否存在
2. **跳过目录备份**：不备份包含checkpoint的目录
3. **加载模型状态**：从checkpoint恢复模型权重、优化器状态、学习率调度器状态
4. **继续训练**：从指定epoch的下一个epoch开始训练

### 输出目录结构
```
results/{model_name}/result_{timestamp}/     # 备份的旧结果
├── {model_name}_ckpt/                      # 权重文件
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── outputs/                                # 训练输出
│   ├── {model_name}_epoch_training_losses.npy
│   ├── {model_name}_epoch_training_scores.npy
│   ├── {model_name}_epoch_test_loss.npy
│   ├── {model_name}_epoch_test_score.npy
│   ├── {model_name}_training_curves.png
│   └── logs/
└── check_predictions/                      # 预测结果
    └── {model_name}_epoch_latest_videos_prediction.pkl

results/{model_name}/result/                # 新的训练结果（或resume时使用现有目录）
├── {model_name}_ckpt/                      # 权重文件
├── outputs/                                # 训练输出
└── check_predictions/                      # 预测结果
```

## 评估

### 基本评估命令
```bash
# 评估单个模型（使用最新/最佳模型）
python eval.py --model ResNetCRNN

# 评估指定epoch的模型
python eval.py --model ResNetCRNN --epoch 50

# 评估所有模型并输出对比
python eval.py --all

# 保存生成的预测结果
python eval.py --model ResNetCRNN --epoch 50 --save-predictions
```

### 评估特性
- **自动预测生成**：如果预测结果文件不存在，会自动生成
- **epoch特定评估**：可以评估任意epoch的模型
- **完整指标计算**：准确率、精确率、召回率、F1分数、混淆矩阵等
- **可视化输出**：自动生成混淆矩阵图和类别性能分析图

### 评估输出目录
```
results/{model_name}/result/outputs/
├── eval_epoch_{N}/                        # 指定epoch的评估结果
│   ├── confusion_matrix.png
│   ├── class_performance.png
│   └── metrics.csv
└── eval_latest/                            # 最新/最佳模型的评估结果
    ├── confusion_matrix.png
    ├── class_performance.png
    └── metrics.csv
```

## 单样本预测

### 基本预测命令
```bash
# ResNetCRNN 单样本预测
python predict.py \
  --model ResNetCRNN \
  --config configs/ResNetCRNN_train.yaml \
  --checkpoint results/ResNetCRNN/result/ResNetCRNN_ckpt/best_model.pth \
  --sample /path/to/video/frames \
  --topk 5

# SwinTransformer-RNN 单样本预测
python predict.py \
  --model swintransformer-RNN \
  --config configs/swintransformer-RNN_train.yaml \
  --checkpoint results/swintransformer-RNN/result/SwinCRNN_ckpt/best_model.pth \
  --sample /path/to/video/frames \
  --topk 5
```

### 预测参数
- `--model`: 模型名称
- `--config`: 配置文件路径
- `--checkpoint`: 训练好的权重文件
- `--sample`: 单个样本的帧目录路径
- `--topk`: 显示前K个预测结果

## 数据预处理

### 视频预处理
```bash
# 基本预处理
python utils/preprocess_videos.py \
  --input_dir ./jpegs_256/UCF-101 \
  --output_dir ./jpegs_256_processed \
  --max_frames 28 \
  --width 256 \
  --height 256

# 恢复中断的预处理
python utils/resume_preprocessing.sh
```

### 数据格式验证
```bash
# 验证单个视频
python utils/test_data_format.py --data_path ./jpegs_256_processed

# 验证多个视频
python utils/test_data_format.py --data_path ./jpegs_256_processed --test_all
```

## 公共组件（utils/common）

### 核心模块
- **label_utils.py**: 标签转换工具（labels2cat/onehot等）
- **data_loaders.py**: 统一数据集类（CRNN/3DCNN/SwinCRNN）
- **model_components.py**: 编码器、解码器与基础模块
- **training_utils.py**: 训练/验证循环、AMP、早停、checkpoint管理

### 工具模块
- **config_loader.py**: YAML配置加载和解析
- **calculate_metrics.py**: 评估指标计算和可视化
- **preprocess_videos.py**: 视频到帧的转换
- **test_data_format.py**: 数据格式验证

## 可视化与结果

### 训练过程
- 训练和验证日志实时输出
- 训练曲线图自动生成
- TensorBoard支持（可选）

### 评估结果
- 混淆矩阵热力图
- 类别性能分析图
- 详细分类报告
- 指标对比表格

## 配置最佳实践

### 1. 配置文件管理
- 为每个实验创建配置文件的副本
- 使用版本控制管理配置文件
- 在配置文件名中添加实验标识（如：`ResNetCRNN_exp001.yaml`）

### 2. 路径配置
- 使用绝对路径避免相对路径问题
- 确保 `data_path` 指向正确的预处理帧目录
- 使用 `configs/data/UCF101actions.pkl` 作为类别文件
- 注意 `save_dir` 格式：`results/{ModelName}/result`

### 3. 训练参数
- 根据GPU显存调整 `batch_size`
- 显存不足时开启 `amp: true`
- 使用 `seed: 42` 保证实验可复现

### 4. 模型选择
- **Conv3D**: 适合时序建模，显存占用高
- **CRNN**: 平衡性能和效率，适合一般应用
- **ResNetCRNN**: 性能优秀，训练稳定，推荐使用
- **SwinTransformer-RNN**: 最新架构，需要较多计算资源

### 5. Resume训练最佳实践
- **定期保存checkpoint**：设置合理的 `save_freq`（如每5个epoch）
- **使用有意义的epoch数**：选择验证性能较好的epoch作为resume点
- **检查checkpoint完整性**：确保checkpoint文件包含完整的训练状态
- **备份重要checkpoint**：手动备份关键epoch的checkpoint文件

## 注意事项

### 环境要求
- Python 3.8+
- PyTorch 1.8+
- CUDA支持（推荐）
- 足够的磁盘空间存储预处理帧

### 数据要求
- 视频帧按 `ActionName/v_ActionName_xxx/` 结构组织
- 每个视频包含28帧（frame000001.jpg 到 frame000028.jpg）
- 帧尺寸建议256x256或224x224

### 性能优化
- 使用SSD存储预处理帧
- 适当设置 `num_workers` 和 `pin_memory`
- 开启混合精度训练（`amp: true`）

### Resume训练注意事项
- **checkpoint文件必须存在**：指定的epoch必须有对应的checkpoint文件
- **配置一致性**：resume时使用的配置文件应与原始训练时一致
- **目录结构完整**：确保checkpoint目录和文件结构完整
- **GPU兼容性**：如果更换GPU，注意显存和CUDA版本兼容性

## 疑难排查

### 常见问题
1. **导入失败**: 确认脚本工作目录为项目根目录
2. **数据加载错误**: 检查 `data_path` 和帧命名格式
3. **显存不足**: 降低 `batch_size` 或开启 `amp: true`
4. **类别不匹配**: 确认 `UCF101actions.pkl` 与数据目录一致

### Resume训练问题
1. **checkpoint不存在**: 检查指定的epoch是否有对应的checkpoint文件
2. **目录被备份**: 确保使用 `--resume` 参数，系统会自动跳过备份
3. **配置不匹配**: 确保resume时使用的配置文件与原始训练一致
4. **显存不足**: 如果更换GPU，可能需要调整batch_size

### 调试工具
- 使用 `utils/test_data_format.py` 验证数据格式
- 检查 `utils/check_progress.py` 查看预处理进度
- 查看训练日志和错误信息
- 使用 `--resume` 参数测试checkpoint加载

### 性能监控
- 使用 `nvidia-smi` 监控GPU使用
- 查看训练曲线图分析收敛情况
- 使用TensorBoard监控训练过程
- 监控checkpoint文件大小和保存时间

## 更新日志

### v2.1 (当前版本)
- **新增Resume训练功能**：支持从指定epoch继续训练
- **智能目录管理**：resume时自动跳过目录备份
- **checkpoint验证**：自动检查checkpoint文件完整性
- **目录管理参数**：新增 `--no_backup` 参数
- 重构为模块化架构
- 统一配置管理
- 自动目录备份和管理
- 完整的评估和预测流程
- 支持SwinTransformer-RNN模型

### v2.0
- 重构为模块化架构
- 统一配置管理
- 自动目录备份和管理
- 完整的评估和预测流程
- 支持SwinTransformer-RNN模型

### v1.0
- 基础模型实现
- 简单的训练脚本
- 手动配置管理
