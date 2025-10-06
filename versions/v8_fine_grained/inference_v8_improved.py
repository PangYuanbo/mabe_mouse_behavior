"""
V8 Improved Inference Script - Fix Kaggle Performance Issues
Key improvements:
1. Lower min_duration and confidence thresholds
2. Overlapping sliding windows for ensemble
3. Smart interval merging
4. Better handling of rare behaviors
"""

import torch
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Adjust path to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from versions.v8_fine_grained.v8_model import V8BehaviorDetector
from versions.v8_fine_grained.v8_dataset import V8Dataset
from versions.v8_fine_grained.submission_utils import predictions_to_intervals
from versions.v8_fine_grained.action_mapping import NUM_ACTIONS


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint"""
    model = V8BehaviorDetector(
        input_dim=config['input_dim'],
        num_actions=NUM_ACTIONS,
        num_mice=config['num_mice'],
        conv_channels=config['conv_channels'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout']
    ).to(device)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[OK] Loaded model from {checkpoint_path}")
    return model


def predict_video_with_overlap(model, video_keypoints, device, sequence_length=100, overlap_ratio=0.5):
    """
    Predict full video using overlapping sliding windows for ensemble
    
    Args:
        model: V8BehaviorDetector
        video_keypoints: [T, D] numpy array
        device: torch device
        sequence_length: window size
        overlap_ratio: overlap between windows (0.5 = 50% overlap)
        
    Returns:
        action_probs: [T, num_actions] ensemble probabilities
        agent_probs: [T, 4] ensemble probabilities 
        target_probs: [T, 4] ensemble probabilities
    """
    T = len(video_keypoints)
    
    # Initialize output arrays
    action_logits_sum = np.zeros((T, NUM_ACTIONS), dtype=np.float32)
    agent_logits_sum = np.zeros((T, 4), dtype=np.float32) 
    target_logits_sum = np.zeros((T, 4), dtype=np.float32)
    counts = np.zeros(T, dtype=np.int32)
    
    # Calculate stride with overlap
    stride = max(1, int(sequence_length * (1 - overlap_ratio)))
    
    print(f"  Using overlapping windows: stride={stride}, overlap={overlap_ratio:.1%}")
    
    # Sliding window prediction with overlap
    for start_idx in range(0, T, stride):
        end_idx = min(start_idx + sequence_length, T)
        window_len = end_idx - start_idx
        
        if window_len < sequence_length // 2:  # Skip very short windows
            continue
            
        # Handle window (pad if necessary)
        if window_len < sequence_length:
            window = np.zeros((sequence_length, video_keypoints.shape[1]), dtype=np.float32)
            window[:window_len] = video_keypoints[start_idx:end_idx]
        else:
            window = video_keypoints[start_idx:end_idx]
        
        # Predict
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_logits, agent_logits, target_logits = model(window_tensor)
        
        # Accumulate logits (better than hard predictions for ensemble)
        action_logits_np = action_logits[0].cpu().numpy()  # [T, num_actions]
        agent_logits_np = agent_logits[0].cpu().numpy()
        target_logits_np = target_logits[0].cpu().numpy()
        
        # Only accumulate valid frames
        valid_len = window_len
        action_logits_sum[start_idx:end_idx] += action_logits_np[:valid_len]
        agent_logits_sum[start_idx:end_idx] += agent_logits_np[:valid_len]
        target_logits_sum[start_idx:end_idx] += target_logits_np[:valid_len]
        counts[start_idx:end_idx] += 1
    
    # Average logits and convert to probabilities
    counts = np.maximum(counts, 1)  # Avoid division by zero
    action_logits_avg = action_logits_sum / counts[:, None]
    agent_logits_avg = agent_logits_sum / counts[:, None]  
    target_logits_avg = target_logits_sum / counts[:, None]
    
    # Convert to probabilities
    from scipy.special import softmax
    action_probs = softmax(action_logits_avg, axis=1)
    agent_probs = softmax(agent_logits_avg, axis=1)
    target_probs = softmax(target_logits_avg, axis=1)
    
    return action_probs, agent_probs, target_probs


def merge_nearby_intervals(intervals, gap_threshold=5):
    """Merge nearby intervals of the same behavior"""
    if not intervals:
        return []
    
    # Sort by start frame
    intervals = sorted(intervals, key=lambda x: x['start_frame'])
    
    merged = []
    current = intervals[0].copy()
    
    for next_interval in intervals[1:]:
        # Check if same behavior and nearby
        same_behavior = (
            current['action_id'] == next_interval['action_id'] and
            current['agent_id'] == next_interval['agent_id'] and
            current['target_id'] == next_interval['target_id']
        )
        
        gap = next_interval['start_frame'] - current['stop_frame'] - 1
        
        if same_behavior and gap <= gap_threshold:
            # Merge intervals
            current['stop_frame'] = next_interval['stop_frame']
            # Weighted average confidence
            w1 = current['stop_frame'] - current['start_frame'] + 1
            w2 = next_interval['stop_frame'] - next_interval['start_frame'] + 1
            current['confidence'] = (current['confidence'] * w1 + next_interval['confidence'] * w2) / (w1 + w2)
        else:
            # Save current and start new
            merged.append(current)
            current = next_interval.copy()
    
    # Add last interval
    merged.append(current)
    
    return merged


def adaptive_min_duration(action_name):
    """Use different minimum durations for different behaviors"""
    # Very brief behaviors
    brief_behaviors = {'bite', 'flinch', 'ejaculate'}
    # Sustained behaviors  
    sustained_behaviors = {'sniff', 'sniffgenital', 'sniffbody', 'sniffface', 'mount', 'huddle'}
    # Medium duration behaviors
    medium_behaviors = {'attack', 'chase', 'escape', 'avoid', 'approach', 'follow'}
    
    if action_name in brief_behaviors:
        return 2  # Very short minimum
    elif action_name in sustained_behaviors:
        return 3  # Short minimum
    elif action_name in medium_behaviors:
        return 4  # Medium minimum
    else:
        return 3  # Default


def generate_improved_submission(
    model,
    data_dir, 
    device,
    output_path='submission_v8_improved.csv',
    base_confidence_threshold=0.3,  # Much lower
    sequence_length=100,
    overlap_ratio=0.5
):
    """
    Generate improved Kaggle submission with better strategies
    """
    data_dir = Path(data_dir)
    
    # Load test metadata
    test_csv = data_dir / 'test.csv'
    if not test_csv.exists():
        raise FileNotFoundError(f"Test metadata not found: {test_csv}")
    
    metadata = pd.read_csv(test_csv)
    print(f"[OK] Found {len(metadata)} test videos")
    
    # Process each video
    all_intervals = []
    row_id = 0
    
    for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing test videos"):
        video_id = row['video_id']
        lab_id = row['lab_id']
        
        tracking_file = data_dir / "test_tracking" / lab_id / f"{video_id}.parquet"
        
        if not tracking_file.exists():
            print(f"[!] Tracking file not found: {tracking_file}")
            continue
        
        # Load and process video
        try:
            tracking_df = pd.read_parquet(tracking_file)
            
            # Standard bodyparts
            standard_bodyparts = [
                'nose', 'ear_left', 'ear_right', 'neck', 
                'hip_left', 'hip_right', 'tail_base'
            ]
            
            tracking_df = tracking_df[tracking_df['bodypart'].isin(standard_bodyparts)]
            
            if len(tracking_df) == 0 or tracking_df['video_frame'].isna().all():
                print(f"[!] No valid tracking data for {video_id}")
                continue
                
            max_frame = tracking_df['video_frame'].max()
            if pd.isna(max_frame):
                continue
                
            num_frames = int(max_frame) + 1
            num_mice = 4
            num_bodyparts = len(standard_bodyparts)
            
            # Pivot and fill keypoints
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
            
            # Add motion features
            keypoints = add_motion_features(keypoints, fps=33.3)
            
            print(f"  Processed {video_id}: {num_frames} frames")
            
            # Predict with ensemble
            action_probs, agent_probs, target_probs = predict_video_with_overlap(
                model, keypoints, device, sequence_length, overlap_ratio
            )
            
            # Convert probabilities to intervals with adaptive thresholds
            intervals = predictions_to_intervals(
                action_preds=action_probs,
                agent_preds=agent_probs, 
                target_preds=target_probs,
                min_duration=2,  # Very low minimum
                confidence_threshold=base_confidence_threshold
            )
            
            print(f"  Initial intervals: {len(intervals)}")
            
            # Apply adaptive minimum durations
            filtered_intervals = []
            for interval in intervals:
                action_name = interval['action']
                min_dur = adaptive_min_duration(action_name)
                duration = interval['stop_frame'] - interval['start_frame'] + 1
                
                if duration >= min_dur:
                    filtered_intervals.append(interval)
            
            print(f"  After adaptive filtering: {len(filtered_intervals)}")
            
            # Merge nearby intervals
            merged_intervals = merge_nearby_intervals(filtered_intervals, gap_threshold=5)
            
            print(f"  After merging: {len(merged_intervals)}")
            
            # Add to submission
            for interval in merged_intervals:
                all_intervals.append({
                    'row_id': row_id,
                    'video_id': video_id,
                    'agent_id': f"mouse{interval['agent_id'] + 1}",
                    'target_id': f"mouse{interval['target_id'] + 1}", 
                    'action': interval['action'],
                    'start_frame': interval['start_frame'],
                    'stop_frame': interval['stop_frame']
                })
                row_id += 1
                
        except Exception as e:
            print(f"[!] Error processing {video_id}: {e}")
            continue
    
    # Create submission DataFrame
    if len(all_intervals) == 0:
        print("[!] WARNING: No predictions generated!")
        submission = pd.DataFrame(columns=[
            'row_id', 'video_id', 'agent_id', 'target_id',
            'action', 'start_frame', 'stop_frame'
        ])
    else:
        submission = pd.DataFrame(all_intervals)
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"\n[OK] Improved submission saved to {output_path}")
    print(f"  Total intervals: {len(submission)}")
    
    if len(submission) > 0:
        print(f"  Unique behaviors: {submission['action'].nunique()}")
        print(f"  Unique videos: {submission['video_id'].nunique()}")
        print("  Top behaviors:")
        for action, count in submission['action'].value_counts().head().items():
            print(f"    {action}: {count}")
    
    return submission


def add_motion_features(keypoints: np.ndarray, fps: float = 33.3) -> np.ndarray:
    """Add speed and acceleration features (same as V8Dataset)"""
    dt = 1.0 / fps
    T, D = keypoints.shape
    
    assert D == 56, f"Expected 56 coords, got {D}"
    
    num_keypoints = D // 2
    coords = keypoints.reshape(T, num_keypoints, 2)
    
    # Velocity
    velocity = np.zeros_like(coords)
    if T > 1:
        velocity[1:] = (coords[1:] - coords[:-1]) / dt
        velocity[0] = velocity[1]
    
    speed = np.sqrt(np.sum(velocity ** 2, axis=2, keepdims=True))
    
    # Acceleration
    acceleration_vec = np.zeros_like(velocity)
    if T > 1:
        acceleration_vec[1:] = (velocity[1:] - velocity[:-1]) / dt
        acceleration_vec[0] = acceleration_vec[1]
    
    acceleration = np.sqrt(np.sum(acceleration_vec ** 2, axis=2, keepdims=True))
    
    # Concatenate
    keypoints_flat = coords.reshape(T, -1)
    speed_flat = speed.squeeze(-1)  
    accel_flat = acceleration.squeeze(-1)
    
    enhanced = np.concatenate([keypoints_flat, speed_flat, accel_flat], axis=1)
    
    return enhanced


def main():
    parser = argparse.ArgumentParser(description='V8 Improved Inference for Better Kaggle Performance')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/v8_5090/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/config_v8_5090.yaml')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default='submission_v8_improved.csv')
    parser.add_argument('--confidence_threshold', type=float, default=0.3)
    parser.add_argument('--overlap_ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Use data_dir from args or config
    data_dir = args.data_dir if args.data_dir else config['data_dir']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    print(f"\n🚀 Running improved inference with:")
    print(f"  - Base confidence threshold: {args.confidence_threshold}")
    print(f"  - Overlap ratio: {args.overlap_ratio}")
    print(f"  - Adaptive minimum durations")
    print(f"  - Smart interval merging")
    
    # Generate improved submission
    submission = generate_improved_submission(
        model=model,
        data_dir=data_dir,
        device=device,
        output_path=args.output,
        base_confidence_threshold=args.confidence_threshold,
        sequence_length=config['sequence_length'],
        overlap_ratio=args.overlap_ratio
    )
    
    print(f"\n[OK] Improved inference complete!")
    print(f"  Submission: {args.output}")
    print(f"  Total predictions: {len(submission)}")
    
    if len(submission) > 0:
        print(f"\n📊 Submission Summary:")
        print(f"  - Videos covered: {submission['video_id'].nunique()}")
        print(f"  - Behaviors detected: {submission['action'].nunique()}")
        print(f"  - Average interval length: {(submission['stop_frame'] - submission['start_frame'] + 1).mean():.1f} frames")


if __name__ == '__main__':
    main()