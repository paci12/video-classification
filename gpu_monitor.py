#!/usr/bin/env python3
"""
GPU监测脚本 - 自动检测空闲GPU并运行训练任务
当检测到空闲GPU时，自动启动指定的训练命令
"""

import subprocess
import time
import logging
import argparse
import os
import signal
import sys
from typing import List, Dict, Optional
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gpu_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUMonitor:
    def __init__(self, 
                 check_interval: int = 60,
                 gpu_memory_threshold: int = 1000,
                 max_concurrent_tasks: int = 2):
        """
        初始化GPU监测器
        
        Args:
            check_interval: 检查间隔（秒）
            gpu_memory_threshold: GPU内存阈值（MB），低于此值认为GPU空闲
            max_concurrent_tasks: 最大并发任务数
        """
        self.check_interval = check_interval
        self.gpu_memory_threshold = gpu_memory_threshold
        self.max_concurrent_tasks = max_concurrent_tasks
        self.running_tasks: List[Dict] = []
        self.task_queue = [
            {
                "name": "ResNetCRNN",
                "command": ["python", "train.py", "--model", "ResNetCRNN", 
                           "--config", "configs/ResNetCRNN_train.yaml"],
                "status": "pending"
            },
            {
                "name": "SwinTransformer-RNN (Resume)",
                "command": ["python", "train.py", "--model", "swintransformer-RNN", 
                           "--config", "configs/swintransformer-RNN_train.yaml", "--resume", "50"],
                "status": "pending"
            }
        ]
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info(f"GPU监测器初始化完成")
        logger.info(f"检查间隔: {check_interval}秒")
        logger.info(f"GPU内存阈值: {gpu_memory_threshold}MB")
        logger.info(f"最大并发任务数: {max_concurrent_tasks}")
        logger.info(f"待执行任务数: {len(self.task_queue)}")

    def signal_handler(self, signum, frame):
        """处理退出信号"""
        logger.info(f"收到信号 {signum}，正在优雅退出...")
        self.cleanup()
        sys.exit(0)

    def get_gpu_info(self) -> List[Dict]:
        """
        获取GPU信息
        
        Returns:
            GPU信息列表，每个GPU包含id, memory_used, memory_total, utilization等信息
        """
        try:
            # 使用gpustat获取GPU信息（JSON格式更可靠）
            result = subprocess.run(
                ["gpustat", "--json"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"gpustat执行失败: {result.stderr}")
                return []
            
            # 解析JSON输出
            data = json.loads(result.stdout)
            gpu_info = []
            
            for gpu in data.get("gpus", []):
                gpu_id = gpu.get("index", 0)
                memory_used = gpu.get("memory.used", 0)
                memory_total = gpu.get("memory.total", 0)
                utilization = gpu.get("utilization.gpu", 0)
                
                # 检查是否有进程在运行
                has_processes = len(gpu.get("processes", [])) > 0
                
                gpu_info.append({
                    "id": gpu_id,
                    "memory_used": memory_used,
                    "memory_total": memory_total,
                    "utilization": utilization,
                    "has_processes": has_processes,
                    "is_idle": memory_used < self.gpu_memory_threshold and utilization < 10 and not has_processes
                })
            
            return gpu_info
            
        except subprocess.TimeoutExpired:
            logger.error("gpustat执行超时")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"解析gpustat JSON输出失败: {e}")
            return []
        except Exception as e:
            logger.error(f"获取GPU信息失败: {e}")
            return []

    def find_idle_gpu(self) -> Optional[int]:
        """
        查找空闲的GPU
        
        Returns:
            空闲GPU的ID，如果没有则返回None
        """
        gpu_info = self.get_gpu_info()
        if not gpu_info:
            return None
            
        for gpu in gpu_info:
            if gpu["is_idle"]:
                logger.info(f"找到空闲GPU {gpu['id']}: "
                          f"内存使用 {gpu['memory_used']}/{gpu['memory_total']}MB, "
                          f"利用率 {gpu['utilization']}%")
                return gpu["id"]
        
        return None

    def start_training_task(self, gpu_id: int, task: Dict) -> bool:
        """
        在指定GPU上启动训练任务
        
        Args:
            gpu_id: GPU ID
            task: 任务信息
            
        Returns:
            是否成功启动
        """
        try:
            # 设置环境变量指定GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # 启动训练进程
            process = subprocess.Popen(
                task["command"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 记录任务信息
            task_info = {
                "name": task["name"],
                "gpu_id": gpu_id,
                "pid": process.pid,
                "process": process,
                "start_time": time.time(),
                "command": " ".join(task["command"])
            }
            
            self.running_tasks.append(task_info)
            task["status"] = "running"
            
            logger.info(f"✅ 启动任务 '{task['name']}' 在GPU {gpu_id} (PID: {process.pid})")
            logger.info(f"命令: {' '.join(task['command'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"启动任务 '{task['name']}' 失败: {e}")
            return False

    def check_running_tasks(self):
        """检查运行中的任务状态"""
        completed_tasks = []
        
        for task in self.running_tasks:
            process = task["process"]
            
            # 检查进程是否还在运行
            if process.poll() is None:
                # 进程还在运行
                continue
            
            # 进程已结束
            return_code = process.returncode
            stdout, stderr = process.communicate()
            
            if return_code == 0:
                logger.info(f"✅ 任务 '{task['name']}' 在GPU {task['gpu_id']} 完成")
            else:
                logger.error(f"❌ 任务 '{task['name']}' 在GPU {task['gpu_id']} 失败 (返回码: {return_code})")
                if stderr:
                    logger.error(f"错误输出: {stderr}")
            
            completed_tasks.append(task)
            
            # 将任务状态重置为pending，以便重新执行
            for queued_task in self.task_queue:
                if queued_task["name"] == task["name"]:
                    queued_task["status"] = "pending"
                    break
        
        # 移除已完成的任务
        for task in completed_tasks:
            self.running_tasks.remove(task)

    def run(self):
        """主运行循环"""
        logger.info("🚀 GPU监测器开始运行...")
        
        try:
            while True:
                # 检查运行中的任务状态
                self.check_running_tasks()
                
                # 如果有空闲GPU且有待执行的任务
                if len(self.running_tasks) < self.max_concurrent_tasks:
                    idle_gpu = self.find_idle_gpu()
                    
                    if idle_gpu is not None:
                        # 查找待执行的任务
                        for task in self.task_queue:
                            if task["status"] == "pending":
                                if self.start_training_task(idle_gpu, task):
                                    break  # 一次只启动一个任务
                
                # 显示当前状态
                self.show_status()
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            logger.info("收到键盘中断，正在退出...")
        except Exception as e:
            logger.error(f"运行过程中发生错误: {e}")
        finally:
            self.cleanup()

    def show_status(self):
        """显示当前状态"""
        logger.info(f"📊 状态更新 - 运行中任务: {len(self.running_tasks)}/{self.max_concurrent_tasks}, "
                   f"待执行任务: {len([t for t in self.task_queue if t['status'] == 'pending'])}")
        
        if self.running_tasks:
            for task in self.running_tasks:
                runtime = time.time() - task["start_time"]
                logger.info(f"  🔄 {task['name']} 在GPU {task['gpu_id']} 运行中 (PID: {task['pid']}, 运行时间: {runtime:.0f}s)")

    def cleanup(self):
        """清理资源"""
        logger.info("🧹 正在清理资源...")
        
        # 终止所有运行中的任务
        for task in self.running_tasks:
            try:
                process = task["process"]
                if process.poll() is None:
                    logger.info(f"终止任务 '{task['name']}' (PID: {task['pid']})")
                    process.terminate()
                    
                    # 等待进程结束
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"强制终止任务 '{task['name']}' (PID: {task['pid']})")
                        process.kill()
                        
            except Exception as e:
                logger.error(f"清理任务 '{task['name']}' 时发生错误: {e}")
        
        logger.info("✅ 资源清理完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPU监测脚本 - 自动运行训练任务")
    parser.add_argument("--check-interval", type=int, default=60,
                       help="检查间隔（秒，默认60）")
    parser.add_argument("--gpu-memory-threshold", type=int, default=1000,
                       help="GPU内存阈值（MB，默认1000）")
    parser.add_argument("--max-concurrent-tasks", type=int, default=2,
                       help="最大并发任务数（默认2）")
    parser.add_argument("--dry-run", action="store_true",
                       help="试运行模式，不实际启动任务")
    
    args = parser.parse_args()
    
    # 检查gpustat是否可用
    try:
        subprocess.run(["gpustat", "--version"], 
                      capture_output=True, check=True, timeout=5)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("❌ gpustat不可用，请先安装gpustat")
        logger.error("安装命令: pip install gpustat")
        logger.error("或者: conda install -c conda-forge gpustat")
        sys.exit(1)
    
    # 检查训练脚本是否存在
    if not os.path.exists("train.py"):
        logger.error("❌ train.py不存在，请确保在正确的目录中运行")
        sys.exit(1)
    
    if not os.path.exists("configs/ResNetCRNN_train.yaml"):
        logger.error("❌ 配置文件不存在，请确保在正确的目录中运行")
        sys.exit(1)
    
    # 创建并运行GPU监测器
    monitor = GPUMonitor(
        check_interval=args.check_interval,
        gpu_memory_threshold=args.gpu_memory_threshold,
        max_concurrent_tasks=args.max_concurrent_tasks
    )
    
    if args.dry_run:
        logger.info("🔍 试运行模式 - 显示GPU信息但不启动任务")
        gpu_info = monitor.get_gpu_info()
        if gpu_info:
            logger.info("GPU信息:")
            for gpu in gpu_info:
                status = "空闲" if gpu["is_idle"] else "忙碌"
                logger.info(f"  GPU {gpu['id']}: {status} "
                          f"(内存: {gpu['memory_used']}/{gpu['memory_total']}MB, "
                          f"利用率: {gpu['utilization']}%)")
        else:
            logger.warning("无法获取GPU信息")
    else:
        monitor.run()

if __name__ == "__main__":
    main()
