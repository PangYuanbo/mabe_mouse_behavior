"""
诊断Kaggle性能差距的分析脚本
分析训练数据与测试数据的分布差异，以及间隔预测质量
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent))

from versions.v8_fine_grained.action_mapping import ID_TO_ACTION, NUM_ACTIONS
from versions.v8_fine_grained.submission_utils import predictions_to_intervals, evaluate_intervals


def analyze_training_data_distribution(data_dir):
    """分析训练数据的行为分布"""
    data_dir = Path(data_dir)
    train_annotation_dir = data_dir / 'train_annotation'
    
    if not train_annotation_dir.exists():
        print(f"训练标注目录未找到: {train_annotation_dir}")
        return None
    
    # 收集所有训练标注数据
    all_intervals = []
    lab_dirs = [d for d in train_annotation_dir.iterdir() if d.is_dir()]
    
    print(f"发现 {len(lab_dirs)} 个实验室数据目录")
    
    for lab_dir in lab_dirs:
        parquet_files = list(lab_dir.glob('*.parquet'))
        print(f"  {lab_dir.name}: {len(parquet_files)} 个视频")
        
        for parquet_file in parquet_files:
            try:
                df = pd.read_parquet(parquet_file)
                all_intervals.append(df)
            except Exception as e:
                print(f"    跳过文件 {parquet_file}: {e}")
    
    if not all_intervals:
        print("未找到有效的训练数据")
        return None
    
    # 合并所有数据
    train_df = pd.concat(all_intervals, ignore_index=True)
    print(f"\n训练数据总计: {len(train_df)} 个间隔")
    
    # 行为分布
    action_counts = train_df['action'].value_counts()
    print(f"\n训练数据行为分布:")
    for action, count in action_counts.head(10).items():
        print(f"  {action}: {count} ({count/len(train_df)*100:.1f}%)")
    
    # 间隔长度分布
    train_df['duration'] = train_df['stop_frame'] - train_df['start_frame'] + 1
    
    print(f"\n间隔长度统计:")
    print(f"  平均长度: {train_df['duration'].mean():.1f} 帧")
    print(f"  中位数长度: {train_df['duration'].median():.1f} 帧")
    print(f"  最小长度: {train_df['duration'].min()} 帧")
    print(f"  最大长度: {train_df['duration'].max()} 帧")
    print(f"  短间隔(<5帧): {(train_df['duration'] < 5).sum()} ({(train_df['duration'] < 5).mean()*100:.1f}%)")
    print(f"  中等间隔(5-30帧): {((train_df['duration'] >= 5) & (train_df['duration'] <= 30)).sum()}")
    print(f"  长间隔(>30帧): {(train_df['duration'] > 30).sum()}")
    
    return {
        'action_distribution': action_counts,
        'duration_stats': train_df['duration'].describe(),
        'total_intervals': len(train_df)
    }


def analyze_submission_quality(submission_path):
    """分析提交文件的质量"""
    if not Path(submission_path).exists():
        print(f"提交文件未找到: {submission_path}")
        return None
    
    submission_df = pd.read_csv(submission_path)
    print(f"\n提交文件分析: {submission_path}")
    print(f"总预测间隔数: {len(submission_df)}")
    
    if len(submission_df) == 0:
        print("⚠️  提交文件为空！这是主要问题之一")
        return None
    
    # 行为分布
    action_counts = submission_df['action'].value_counts()
    print(f"\n提交的行为分布:")
    for action, count in action_counts.items():
        print(f"  {action}: {count} ({count/len(submission_df)*100:.1f}%)")
    
    # 间隔长度分析
    submission_df['duration'] = submission_df['stop_frame'] - submission_df['start_frame'] + 1
    
    print(f"\n提交间隔长度统计:")
    print(f"  平均长度: {submission_df['duration'].mean():.1f} 帧")
    print(f"  中位数长度: {submission_df['duration'].median():.1f} 帧")
    print(f"  最小长度: {submission_df['duration'].min()} 帧")
    print(f"  最大长度: {submission_df['duration'].max()} 帧")
    
    # 视频覆盖度
    unique_videos = submission_df['video_id'].nunique()
    print(f"\n视频覆盖:")
    print(f"  涉及视频数: {unique_videos}")
    
    return {
        'total_predictions': len(submission_df),
        'action_distribution': action_counts,
        'duration_stats': submission_df['duration'].describe(),
        'unique_videos': unique_videos
    }


def simulate_interval_conversion():
    """模拟帧级预测到间隔转换的影响"""
    print("\n=== 模拟间隔转换分析 ===")
    
    # 创建模拟数据 - 碎片化预测
    T = 1000
    action_preds = np.zeros(T, dtype=int)
    
    # 模拟一些真实行为间隔
    true_intervals = [
        (100, 150, 12),  # attack: 50帧
        (200, 210, 12),  # attack: 10帧  
        (300, 305, 13),  # chase: 5帧
        (400, 420, 19),  # avoid: 20帧
    ]
    
    for start, end, action in true_intervals:
        action_preds[start:end+1] = action
    
    # 添加噪声 - 模拟预测不稳定
    for start, end, action in true_intervals:
        # 随机移除一些帧
        noise_mask = np.random.random(end-start+1) < 0.2  # 20%噪声
        noise_indices = np.where(noise_mask)[0] + start
        action_preds[noise_indices] = 0
    
    agent_preds = np.random.randint(0, 4, T)
    target_preds = np.random.randint(0, 4, T)
    
    # 测试不同参数的影响
    min_durations = [1, 3, 5, 10]
    confidence_thresholds = [0.3, 0.5, 0.7, 0.9]
    
    print(f"模拟数据: {len(true_intervals)} 个真实间隔")
    
    for min_dur in min_durations:
        for conf_thresh in confidence_thresholds:
            intervals = predictions_to_intervals(
                action_preds, agent_preds, target_preds,
                min_duration=min_dur, confidence_threshold=conf_thresh
            )
            
            # 过滤掉background
            intervals = [i for i in intervals if i['action_id'] > 0]
            
            print(f"min_duration={min_dur:2d}, confidence={conf_thresh:.1f}: {len(intervals):2d} 间隔")


def test_interval_postprocessing():
    """测试不同后处理策略"""
    print("\n=== 间隔后处理策略测试 ===")
    
    # 模拟碎片化预测
    intervals = [
        {'action_id': 12, 'agent_id': 0, 'target_id': 1, 'start_frame': 100, 'stop_frame': 105, 'confidence': 0.8},
        {'action_id': 12, 'agent_id': 0, 'target_id': 1, 'start_frame': 107, 'stop_frame': 110, 'confidence': 0.7},  # 小间隙
        {'action_id': 12, 'agent_id': 0, 'target_id': 1, 'start_frame': 115, 'stop_frame': 120, 'confidence': 0.6},  # 较大间隙
        
        {'action_id': 13, 'agent_id': 1, 'target_id': 2, 'start_frame': 200, 'stop_frame': 201, 'confidence': 0.9},  # 很短
        {'action_id': 13, 'agent_id': 1, 'target_id': 2, 'start_frame': 203, 'stop_frame': 208, 'confidence': 0.8},
    ]
    
    print(f"原始间隔: {len(intervals)}")
    for i, interval in enumerate(intervals):
        duration = interval['stop_frame'] - interval['start_frame'] + 1
        print(f"  #{i}: {ID_TO_ACTION[interval['action_id']]} 帧{interval['start_frame']}-{interval['stop_frame']} ({duration}帧)")
    
    # 策略1: 基本过滤 (当前方法)
    basic_filtered = [i for i in intervals if (i['stop_frame'] - i['start_frame'] + 1) >= 5]
    print(f"\n基本过滤(min_duration=5): {len(basic_filtered)} 间隔")
    
    # 策略2: 智能合并
    merged = merge_nearby_intervals(intervals.copy(), gap_threshold=5)
    print(f"智能合并(gap<=5): {len(merged)} 间隔")
    for i, interval in enumerate(merged):
        duration = interval['stop_frame'] - interval['start_frame'] + 1
        print(f"  #{i}: {ID_TO_ACTION[interval['action_id']]} 帧{interval['start_frame']}-{interval['stop_frame']} ({duration}帧)")


def merge_nearby_intervals(intervals, gap_threshold=5):
    """合并相近的同类间隔"""
    if not intervals:
        return []
    
    # 按开始帧排序
    intervals = sorted(intervals, key=lambda x: x['start_frame'])
    
    merged = []
    current = intervals[0].copy()
    
    for next_interval in intervals[1:]:
        # 检查是否为同类间隔且距离较近
        same_behavior = (
            current['action_id'] == next_interval['action_id'] and
            current['agent_id'] == next_interval['agent_id'] and
            current['target_id'] == next_interval['target_id']
        )
        
        gap = next_interval['start_frame'] - current['stop_frame'] - 1
        
        if same_behavior and gap <= gap_threshold:
            # 合并
            current['stop_frame'] = next_interval['stop_frame']
            # 权重平均置信度
            w1 = current['stop_frame'] - current['start_frame'] + 1
            w2 = next_interval['stop_frame'] - next_interval['start_frame'] + 1
            current['confidence'] = (current['confidence'] * w1 + next_interval['confidence'] * w2) / (w1 + w2)
        else:
            # 不能合并，保存当前间隔
            merged.append(current)
            current = next_interval.copy()
    
    # 添加最后一个间隔
    merged.append(current)
    
    return merged


def recommend_improvements():
    """给出改进建议"""
    print("\n" + "="*60)
    print("🎯 Kaggle得分改进建议")
    print("="*60)
    
    print("\n1. 📊 数据分析和验证")
    print("   - 使用时间分割而非随机分割创建验证集")
    print("   - 分析训练集中不同行为的时间分布特征")
    print("   - 检查测试集视频是否来自不同时间段/实验条件")
    
    print("\n2. 🔧 间隔预测优化")
    print("   - 降低min_duration到3或更小")
    print("   - 降低confidence_threshold到0.3或更小") 
    print("   - 实现智能间隔合并，处理碎片化预测")
    
    print("\n3. 🧠 模型推理改进")
    print("   - 使用重叠滑动窗口(50%重叠)进行ensemble")
    print("   - 实现测试时数据增强(TTA)")
    print("   - 对短间隔使用不同的置信度阈值")
    
    print("\n4. 📈 后处理策略")
    print("   - 合并间隔距离<=5帧的同类行为")
    print("   - 对不同行为类别使用不同的最小长度要求")
    print("   - 基于行为类别的自适应置信度阈值")
    
    print("\n5. 🎛️ 立即可尝试的快速修复")
    print("   - 修改inference_v8.py中的min_duration=3")
    print("   - 在sliding window中添加50%重叠")
    print("   - 实现间隔后处理合并")


def main():
    print("🔍 Kaggle性能诊断分析")
    print("="*60)
    
    # 分析训练数据
    data_dir = "C:/Users/aaron/PycharmProjects/mabe_mouse_behavior/data/kaggle"
    train_stats = analyze_training_data_distribution(data_dir)
    
    # 分析提交文件
    submission_path = "submission_v8_local_test.csv"
    submission_stats = analyze_submission_quality(submission_path)
    
    # 模拟测试
    simulate_interval_conversion()
    test_interval_postprocessing()
    
    # 给出建议
    recommend_improvements()


if __name__ == "__main__":
    main()