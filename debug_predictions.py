"""
调试脚本 - 检查V8模型的预测分布
找出为什么只预测attack而不是28种行为
"""

import torch
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from versions.v8_fine_grained.v8_model import V8BehaviorDetector
from versions.v8_fine_grained.action_mapping import NUM_ACTIONS, ID_TO_ACTION
from inference_v8_improved import load_model, add_motion_features


def debug_model_predictions():
    """调试模型预测分布"""
    
    # 加载配置和模型
    config_path = 'configs/config_v8_5090.yaml'
    checkpoint_path = 'checkpoints/v8_5090/best_model.pth'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(checkpoint_path, config, device)
    
    print(f"模型加载完成，NUM_ACTIONS = {NUM_ACTIONS}")
    print(f"行为映射: {list(ID_TO_ACTION.items())}")
    
    # 加载测试数据
    data_dir = Path(config['data_dir'])
    test_csv = data_dir / 'test.csv'
    metadata = pd.read_csv(test_csv)
    
    print(f"\n处理测试视频...")
    
    for idx, row in metadata.iterrows():
        video_id = row['video_id']
        lab_id = row['lab_id']
        
        tracking_file = data_dir / "test_tracking" / lab_id / f"{video_id}.parquet"
        
        if not tracking_file.exists():
            print(f"找不到文件: {tracking_file}")
            continue
            
        # 处理数据
        tracking_df = pd.read_parquet(tracking_file)
        
        standard_bodyparts = [
            'nose', 'ear_left', 'ear_right', 'neck',
            'hip_left', 'hip_right', 'tail_base'
        ]
        
        tracking_df = tracking_df[tracking_df['bodypart'].isin(standard_bodyparts)]
        
        if len(tracking_df) == 0:
            continue
            
        max_frame = tracking_df['video_frame'].max()
        num_frames = int(max_frame) + 1
        num_mice = 4
        num_bodyparts = len(standard_bodyparts)
        
        # 构建关键点数据
        x_pivot = tracking_df.pivot_table(
            index='video_frame',
            columns=['mouse_id', 'bodypart'],
            values='x',
            aggfunc='first'
        )
        y_pivot = tracking_df.pivot_table(
            index='video_frame',
            columns=['mouse_id', 'bodypart'],
            values='y',
            aggfunc='first'
        )
        
        keypoints_raw = np.zeros((num_frames, num_mice * num_bodyparts * 2), dtype=np.float32)
        
        for mouse_id in range(1, 5):
            for bp_idx, bodypart in enumerate(standard_bodyparts):
                if (mouse_id, bodypart) in x_pivot.columns:
                    frames = x_pivot.index.values.astype(int)
                    x_vals = x_pivot[(mouse_id, bodypart)].values
                    y_vals = y_pivot[(mouse_id, bodypart)].values
                    
                    base_idx = (mouse_id - 1) * num_bodyparts * 2 + bp_idx * 2
                    keypoints_raw[frames, base_idx] = x_vals
                    keypoints_raw[frames, base_idx + 1] = y_vals
        
        keypoints = np.nan_to_num(keypoints_raw, nan=0.0)
        keypoints = add_motion_features(keypoints, fps=33.3)
        
        print(f"视频 {video_id}: {num_frames} 帧, 特征维度: {keypoints.shape[1]}")
        
        # 预测一个窗口来查看分布
        sequence_length = config['sequence_length']
        start_idx = 1000  # 从中间开始取样
        end_idx = min(start_idx + sequence_length, num_frames)
        
        if end_idx - start_idx >= 50:  # 确保有足够的帧
            window = keypoints[start_idx:end_idx]
            
            if len(window) < sequence_length:
                padded_window = np.zeros((sequence_length, keypoints.shape[1]), dtype=np.float32)
                padded_window[:len(window)] = window
                window = padded_window
            
            window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_logits, agent_logits, target_logits = model(window_tensor)
            
            # 分析预测分布
            action_probs = torch.softmax(action_logits[0], dim=-1).cpu().numpy()  # [T, num_actions]
            agent_probs = torch.softmax(agent_logits[0], dim=-1).cpu().numpy()    # [T, 4]
            target_probs = torch.softmax(target_logits[0], dim=-1).cpu().numpy()  # [T, 4]
            
            print(f"\n=== 预测分析 (帧 {start_idx}-{end_idx}) ===")
            
            # 检查action预测分布
            print("\n🎯 Action预测统计:")
            print(f"Action logits shape: {action_logits.shape}")
            print(f"Action probs shape: {action_probs.shape}")
            
            # 统计每个行为的平均概率
            avg_probs = np.mean(action_probs, axis=0)
            max_probs = np.max(action_probs, axis=0)
            
            print(f"\n各行为平均概率 (前10个):")
            for i in range(min(10, len(avg_probs))):
                action_name = ID_TO_ACTION.get(i, f'unknown_{i}')
                print(f"  {i:2d} {action_name:15s}: avg={avg_probs[i]:.4f}, max={max_probs[i]:.4f}")
            
            print(f"\n各行为平均概率 (后面的):")
            for i in range(10, len(avg_probs)):
                action_name = ID_TO_ACTION.get(i, f'unknown_{i}')
                if avg_probs[i] > 0.01:  # 只显示概率>1%的
                    print(f"  {i:2d} {action_name:15s}: avg={avg_probs[i]:.4f}, max={max_probs[i]:.4f}")
            
            # 找出最大概率的预测
            predicted_actions = np.argmax(action_probs, axis=1)
            unique_preds, counts = np.unique(predicted_actions, return_counts=True)
            
            print(f"\n预测结果统计:")
            for pred_id, count in zip(unique_preds, counts):
                action_name = ID_TO_ACTION.get(pred_id, f'unknown_{pred_id}')
                percentage = count / len(predicted_actions) * 100
                print(f"  {pred_id:2d} {action_name:15s}: {count:4d} 帧 ({percentage:5.1f}%)")
            
            # 检查是否有非背景预测
            non_background = predicted_actions[predicted_actions != 0]
            if len(non_background) > 0:
                print(f"\n非背景预测: {len(non_background)} 帧")
                unique_nb, counts_nb = np.unique(non_background, return_counts=True)
                for pred_id, count in zip(unique_nb, counts_nb):
                    action_name = ID_TO_ACTION.get(pred_id, f'unknown_{pred_id}')
                    print(f"  {pred_id:2d} {action_name:15s}: {count:4d} 帧")
            else:
                print(f"\n⚠️  没有非背景预测！全部是背景类(id=0)")
            
            # 检查agent和target预测
            predicted_agents = np.argmax(agent_probs, axis=1)
            predicted_targets = np.argmax(target_probs, axis=1)
            
            print(f"\nAgent预测分布: {np.bincount(predicted_agents)}")
            print(f"Target预测分布: {np.bincount(predicted_targets)}")
            
            # 检查置信度分布
            max_action_probs = np.max(action_probs, axis=1)
            print(f"\n置信度统计:")
            print(f"  平均最大概率: {np.mean(max_action_probs):.4f}")
            print(f"  最大概率分布:")
            print(f"    >0.9: {np.sum(max_action_probs > 0.9)} 帧")
            print(f"    >0.8: {np.sum(max_action_probs > 0.8)} 帧") 
            print(f"    >0.7: {np.sum(max_action_probs > 0.7)} 帧")
            print(f"    >0.6: {np.sum(max_action_probs > 0.6)} 帧")
            print(f"    >0.5: {np.sum(max_action_probs > 0.5)} 帧")
            
            # 找出最有信心的非背景预测
            non_bg_mask = predicted_actions != 0
            if np.any(non_bg_mask):
                non_bg_confidences = max_action_probs[non_bg_mask]
                non_bg_actions = predicted_actions[non_bg_mask]
                
                # 按置信度排序
                sorted_indices = np.argsort(non_bg_confidences)[::-1]
                print(f"\n最有信心的非背景预测 (前5个):")
                for i in range(min(5, len(sorted_indices))):
                    idx = sorted_indices[i]
                    frame_idx = np.where(non_bg_mask)[0][idx]
                    action_id = non_bg_actions[idx]
                    confidence = non_bg_confidences[idx]
                    action_name = ID_TO_ACTION.get(action_id, f'unknown_{action_id}')
                    print(f"  帧{frame_idx}: {action_name} (id={action_id}) 置信度={confidence:.4f}")
            
            break  # 只分析第一个视频


if __name__ == "__main__":
    debug_model_predictions()