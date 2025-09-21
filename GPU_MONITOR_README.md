# 🚀 GPU监测脚本使用指南

## 📋 功能概述

`gpu_monitor.py` 是一个智能GPU监测脚本，能够：
- 🔍 自动检测空闲GPU
- 🎯 智能分配训练任务到空闲GPU
- 📊 实时监控任务状态
- 🔄 自动重启失败的任务
- 📝 详细记录运行日志

### 🚀 为什么选择gpustat？

脚本使用 `gpustat` 而不是 `nvidia-smi`，因为：
- **更清晰**: 输出格式更易读，信息更直观
- **更轻量**: 资源占用更少，响应更快
- **更友好**: 自动处理单位转换，显示更人性化
- **更稳定**: 错误处理更好，兼容性更强

## 🎯 预设任务

脚本预设了两个训练任务：

1. **ResNetCRNN训练**
   ```bash
   python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml
   ```

2. **SwinTransformer-RNN继续训练**
   ```bash
   python train.py --model swintransformer-RNN --config configs/swintransformer-RNN_train.yaml --resume 50
   ```

## 🚀 快速开始

### 1. 基本使用

```bash
# 启动GPU监测器（默认设置）
python gpu_monitor.py

# 试运行模式（只显示GPU信息，不启动任务）
python gpu_monitor.py --dry-run
```

### 2. 自定义参数

```bash
# 自定义检查间隔和GPU阈值
python gpu_monitor.py --check-interval 30 --gpu-memory-threshold 500

# 限制最大并发任务数
python gpu_monitor.py --max-concurrent-tasks 1

# 组合使用
python gpu_monitor.py \
    --check-interval 45 \
    --gpu-memory-threshold 800 \
    --max-concurrent-tasks 3
```

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--check-interval` | 60 | 检查GPU状态的间隔（秒） |
| `--gpu-memory-threshold` | 1000 | GPU内存阈值（MB），低于此值认为GPU空闲 |
| `--max-concurrent-tasks` | 2 | 最大并发运行的任务数 |
| `--dry-run` | False | 试运行模式，不实际启动任务 |

## 🔍 GPU空闲判断标准

GPU被认为是空闲的条件：
- 内存使用 < `gpu_memory_threshold`（默认1000MB）
- GPU利用率 < 10%
- 没有进程在运行

## 📊 运行状态

### 启动时显示
```
🚀 GPU监测器开始运行...
📊 状态更新 - 运行中任务: 0/2, 待执行任务: 2
```

### 找到空闲GPU时
```
✅ 启动任务 'ResNetCRNN' 在GPU 0 (PID: 12345)
命令: python train.py --model ResNetCRNN --config configs/ResNetCRNN_train.yaml
```

### 任务完成时
```
✅ 任务 'ResNetCRNN' 在GPU 0 完成
```

### 任务失败时
```
❌ 任务 'ResNetCRNN' 在GPU 0 失败 (返回码: 1)
错误输出: [错误详情]
```

## 📁 日志文件

脚本会自动创建 `gpu_monitor.log` 文件，记录所有运行信息：

```bash
# 查看实时日志
tail -f gpu_monitor.log

# 查看最近的日志
tail -100 gpu_monitor.log
```

## 🛠️ 高级用法

### 1. 后台运行

```bash
# 使用nohup后台运行
nohup python gpu_monitor.py > gpu_monitor.out 2>&1 &

# 查看进程
ps aux | grep gpu_monitor

# 停止进程
kill [PID]
```

### 2. 使用screen/tmux

```bash
# 创建新的screen会话
screen -S gpu_monitor

# 在screen中运行
python gpu_monitor.py

# 分离会话：Ctrl+A, D
# 重新连接：screen -r gpu_monitor
```

### 3. 系统服务（可选）

创建systemd服务文件 `/etc/systemd/system/gpu-monitor.service`：

```ini
[Unit]
Description=GPU Monitor Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/video-classification
ExecStart=/usr/bin/python3 gpu_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl enable gpu-monitor
sudo systemctl start gpu-monitor
sudo systemctl status gpu-monitor
```

## 🔧 故障排除

### 1. gpustat不可用
```bash
# 检查gpustat是否安装
gpustat --version

# 安装gpustat
pip install gpustat

# 或者使用conda
conda install -c conda-forge gpustat

# 如果pip安装失败，可能需要先安装NVIDIA驱动
nvidia-smi --version
```

### 2. 权限问题
```bash
# 确保脚本有执行权限
chmod +x gpu_monitor.py

# 检查CUDA环境变量
echo $CUDA_HOME
echo $PATH | grep cuda
```

### 3. 训练脚本路径问题
```bash
# 确保在正确的目录中运行
pwd
ls -la train.py
ls -la configs/
```

### 4. 查看详细错误
```bash
# 查看stderr输出
python gpu_monitor.py 2>&1 | tee gpu_monitor_detailed.log
```

## 📈 性能优化建议

### 1. 检查间隔
- **开发环境**: 30-60秒
- **生产环境**: 60-120秒
- **高负载环境**: 120-300秒

### 2. GPU内存阈值
- **小模型**: 500-800MB
- **中等模型**: 800-1500MB
- **大模型**: 1500-3000MB

### 3. 并发任务数
- **单GPU**: 1
- **多GPU**: 根据GPU数量调整
- **内存受限**: 减少并发数

## 🔄 任务管理

### 添加新任务
编辑 `gpu_monitor.py` 中的 `task_queue`：

```python
self.task_queue = [
    # 现有任务...
    {
        "name": "新任务名称",
        "command": ["python", "train.py", "--model", "新模型", "--config", "配置文件"],
        "status": "pending"
    }
]
```

### 修改任务参数
```python
# 修改ResNetCRNN的配置
{
    "name": "ResNetCRNN (自定义)",
    "command": ["python", "train.py", "--model", "ResNetCRNN", 
               "--config", "configs/ResNetCRNN_custom.yaml"],
    "status": "pending"
}
```

## 🎉 使用示例

### 示例1：基本监控
```bash
cd /path/to/video-classification
python gpu_monitor.py
```

### 示例2：快速检查GPU状态
```bash
python gpu_monitor.py --dry-run
```

### 示例3：高频率监控
```bash
python gpu_monitor.py --check-interval 30 --gpu-memory-threshold 500
```

### 示例4：单任务模式
```bash
python gpu_monitor.py --max-concurrent-tasks 1
```

## 📞 技术支持

如果遇到问题，请检查：
1. ✅ NVIDIA驱动是否正确安装
2. ✅ CUDA环境是否正确配置
3. ✅ 训练脚本和配置文件是否存在
4. ✅ 日志文件中的错误信息

---

**祝你的训练任务顺利运行！🚀**
