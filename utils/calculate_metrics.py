import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib后端和字体
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_predictions(model_name="ResNetCRNN"):
    """加载预测结果"""
    pred_file = f'results/{model_name}/result/check_predictions/{model_name}_videos_prediction.pkl'
    if os.path.exists(pred_file):
        with open(pred_file, 'rb') as f:
            predictions_df = pickle.load(f)
        print(f"加载预测结果: {len(predictions_df)} 个样本")
        return predictions_df
    else:
        print(f"预测结果文件不存在: {pred_file}")
        return None

def calculate_metrics(y_true, y_pred, y_prob=None):
    """计算各种评估指标"""
    metrics = {}
    
    # 基础分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # 多分类指标 (UCF101有101个类别)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 如果有概率值，计算AUC
    if y_prob is not None:
        try:
            # 对于多分类，计算每个类别的one-vs-rest AUC的平均值
            if len(np.unique(y_true)) > 2:
                metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            else:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob)
        except Exception as e:
            print(f"AUC计算失败: {e}")
            metrics['auc_ovr'] = None
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, action_names, save_path='confusion_matrix.png'):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', 
                xticklabels=False, yticklabels=False)
    plt.title('Confusion Matrix (UCF101 Actions)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {save_path}")

def plot_class_performance(y_true, y_pred, action_names, save_path='class_performance.png'):
    """绘制每个类别的性能"""
    # 添加调试信息
    print(f"y_true类型: {type(y_true)}, 形状: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}")
    print(f"y_pred类型: {type(y_pred)}, 形状: {y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred)}")
    print(f"y_true前5个值: {y_true[:5]}")
    print(f"y_pred前5个值: {y_pred[:5]}")
    print(f"y_true唯一值: {np.unique(y_true)}")
    print(f"y_pred唯一值: {np.unique(y_pred)}")
    print(f"action_names长度: {len(action_names)}")
    
    # 计算每个类别的指标
    class_metrics = []
    for i in range(len(action_names)):
        # 确保标签是数值类型进行比较
        if isinstance(y_true[0], str):
            # 如果是字符串标签，需要转换为数值
            class_mask = (y_true == action_names[i])
        else:
            # 如果是数值标签
            class_mask = (y_true == i)
            
        if np.sum(class_mask) > 0:
            try:
                class_precision = precision_score(y_true == i, y_pred == i, zero_division=0)
                class_recall = recall_score(y_true == i, y_pred == i, zero_division=0)
                class_f1 = f1_score(y_true == i, y_pred == i, zero_division=0)
                class_accuracy = accuracy_score(y_true == i, y_pred == i)
                
                class_metrics.append({
                    'class_id': i,
                    'class_name': action_names[i],
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1': class_f1,
                    'accuracy': class_accuracy,
                    'support': np.sum(class_mask)
                })
                print(f"类别 {i} ({action_names[i]}): 支持度={np.sum(class_mask)}")
            except Exception as e:
                print(f"计算类别 {i} 指标时出错: {e}")
    
    print(f"成功计算的类别数量: {len(class_metrics)}")
    
    # 转换为DataFrame并排序
    class_df = pd.DataFrame(class_metrics)
    
    # 添加调试信息
    print(f"DataFrame列名: {class_df.columns.tolist()}")
    print(f"DataFrame形状: {class_df.shape}")
    if not class_df.empty:
        print(f"前几行数据:")
        print(class_df.head())
        
        class_df = class_df.sort_values('f1', ascending=False)
        
        # 绘制前20个表现最好的类别
        top_20 = class_df.head(20)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 前20个类别的F1分数
        axes[0, 0].barh(range(len(top_20)), top_20['f1'])
        axes[0, 0].set_yticks(range(len(top_20)))
        axes[0, 0].set_yticklabels([f"{i}: {name[:20]}..." if len(name) > 20 else f"{i}: {name}" 
                                     for i, name in zip(top_20['class_id'], top_20['class_name'])])
        axes[0, 0].set_xlabel('F1 Score')
        axes[0, 0].set_title('Top 20 Classes by F1 Score')
        axes[0, 0].set_xlim(0, 1)
        
        # 前20个类别的准确率
        axes[0, 1].barh(range(len(top_20)), top_20['accuracy'])
        axes[0, 1].set_yticks(range(len(top_20)))
        axes[0, 1].set_yticklabels([f"{i}: {name[:20]}..." if len(name) > 20 else f"{i}: {name}" 
                                     for i, name in zip(top_20['class_id'], top_20['class_name'])])
        axes[0, 1].set_xlabel('Accuracy')
        axes[0, 1].set_title('Top 20 Classes by Accuracy')
        axes[0, 1].set_xlim(0, 1)
        
        # 前20个类别的召回率
        axes[1, 0].barh(range(len(top_20)), top_20['recall'])
        axes[1, 0].set_yticks(range(len(top_20)))
        axes[1, 0].set_yticklabels([f"{i}: {name[:20]}..." if len(name) > 20 else f"{i}: {name}" 
                                     for i, name in zip(top_20['class_id'], top_20['class_name'])])
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_title('Top 20 Classes by Recall')
        axes[1, 0].set_xlim(0, 1)
        
        # 前20个类别的精确率
        axes[1, 1].barh(range(len(top_20)), top_20['precision'])
        axes[1, 1].set_yticks(range(len(top_20)))
        axes[1, 1].set_yticklabels([f"{i}: {name[:20]}..." if len(name) > 20 else f"{i}: {name}" 
                                     for i, name in zip(top_20['class_id'], top_20['class_name'])])
        axes[1, 1].set_xlabel('Precision')
        axes[1, 1].set_title('Top 20 Classes by Precision')
        axes[1, 1].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"类别性能分析已保存: {save_path}")
    else:
        print("警告: DataFrame为空，无法绘制类别性能分析")
    
    return class_df

def main():
    """主函数"""
    print("开始计算CRNN模型的全面评估指标...")
    
    # 加载预测结果
    predictions_df = load_predictions("ResNetCRNN")
    if predictions_df is None:
        return
    
    # 加载动作名称
    action_names_file = f'models/{model_name}/UCF101actions.pkl'
    if os.path.exists(action_names_file):
        with open(action_names_file, 'rb') as f:
            action_names = pickle.load(f)
        print(f"加载动作名称: {len(action_names)} 个类别")
    else:
        print("动作名称文件不存在，使用数字标签")
        action_names = [f"Class_{i}" for i in range(101)]
    
    # 检查数据列
    print(f"预测结果列名: {predictions_df.columns.tolist()}")
    
    # 根据实际列名调整
    if 'y' in predictions_df.columns and 'y_pred' in predictions_df.columns:
        y_true = predictions_df['y'].values
        y_pred = predictions_df['y_pred'].values
        
        # 检查是否有概率值
        prob_cols = [col for col in predictions_df.columns if 'prob' in col.lower() or 'confidence' in col.lower()]
        y_prob = None
        if prob_cols:
            y_prob = predictions_df[prob_cols[0]].values
            print(f"使用概率列: {prob_cols[0]}")
        
        print(f"真实标签范围: {y_true.min()} - {y_true.max()}")
        print(f"预测标签范围: {y_pred.min()} - {y_pred.max()}")
        print(f"样本数量: {len(y_true)}")
        
        # 计算指标
        print("\n计算评估指标...")
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # 打印结果
        print("\n" + "="*60)
        print("CRNN模型全面评估结果")
        print("="*60)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"精确率 (Precision) - Macro: {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)")
        print(f"精确率 (Precision) - Weighted: {metrics['precision_weighted']:.4f} ({metrics['precision_weighted']*100:.2f}%)")
        print(f"召回率 (Recall) - Macro: {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)")
        print(f"召回率 (Recall) - Weighted: {metrics['recall_weighted']:.4f} ({metrics['recall_weighted']*100:.2f}%)")
        print(f"F1分数 - Macro: {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
        print(f"F1分数 - Weighted: {metrics['f1_weighted']:.4f} ({metrics['f1_weighted']*100:.2f}%)")
        
        if metrics.get('auc_ovr') is not None:
            print(f"AUC (One-vs-Rest): {metrics['auc_ovr']:.4f} ({metrics['auc_ovr']*100:.2f}%)")
        
        # 生成详细分类报告
        print("\n" + "="*60)
        print("详细分类报告")
        print("="*60)
        report = classification_report(y_true, y_pred, target_names=action_names[:101], 
                                     digits=4, zero_division=0)
        print(report)
        
        # 保存报告到文件
        with open('classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("CRNN模型分类报告\n")
            f.write("="*60 + "\n")
            f.write(report)
        print("\n分类报告已保存到: classification_report.txt")
        
        # 绘制混淆矩阵
        print("\n绘制混淆矩阵...")
        plot_confusion_matrix(y_true, y_pred, action_names)
        
        # 绘制类别性能分析
        print("绘制类别性能分析...")
        class_df = plot_class_performance(y_true, y_pred, action_names)
        
        # 保存类别性能数据
        class_df.to_csv('class_performance_metrics.csv', index=False)
        print("类别性能指标已保存到: class_performance_metrics.csv")
        
        # 保存总体指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('overall_metrics.csv', index=False)
        print("总体指标已保存到: overall_metrics.csv")
        
    else:
        print("预测结果文件格式不正确，请检查列名")
        print("期望的列名: y, y_pred")
        print("实际的列名:", predictions_df.columns.tolist())

if __name__ == "__main__":
    main() 