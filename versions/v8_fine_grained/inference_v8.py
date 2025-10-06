"""
V8 Inference Script - Generate Kaggle Submission
Usage: python versions/v8_fine_grained/inference_v8.py --checkpoint checkpoints/v8_5090/best_model.pth
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


def predict_video(model, video_keypoints, device, sequence_length=100):
    """
    Predict full video using sliding window

    Args:
        model: V8BehaviorDetector
        video_keypoints: [T, D] numpy array
        device: torch device
        sequence_length: window size

    Returns:
        action_preds: [T] predicted action IDs
        agent_preds: [T] predicted agent IDs
        target_preds: [T] predicted target IDs
    """
    T = len(video_keypoints)

    # Initialize output arrays
    action_preds = np.zeros(T, dtype=np.int64)
    agent_preds = np.zeros(T, dtype=np.int64)
    target_preds = np.zeros(T, dtype=np.int64)
    counts = np.zeros(T, dtype=np.int64)  # For averaging overlapping windows

    # Sliding window prediction
    stride = sequence_length  # No overlap for inference
    for start_idx in range(0, T, stride):
        end_idx = min(start_idx + sequence_length, T)
        window_len = end_idx - start_idx

        # Handle last window (might be shorter)
        if window_len < sequence_length:
            # Pad to sequence_length
            window = np.zeros((sequence_length, video_keypoints.shape[1]), dtype=np.float32)
            window[:window_len] = video_keypoints[start_idx:end_idx]
        else:
            window = video_keypoints[start_idx:end_idx]

        # Predict
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(device)  # [1, T, D]

        with torch.no_grad():
            action_logits, agent_logits, target_logits = model(window_tensor)

        # Get predictions
        action_pred = action_logits[0].argmax(dim=-1).cpu().numpy()  # [T]
        agent_pred = agent_logits[0].argmax(dim=-1).cpu().numpy()
        target_pred = target_logits[0].argmax(dim=-1).cpu().numpy()

        # Accumulate (only valid frames)
        valid_len = window_len
        action_preds[start_idx:end_idx] += action_pred[:valid_len]
        agent_preds[start_idx:end_idx] += agent_pred[:valid_len]
        target_preds[start_idx:end_idx] += target_pred[:valid_len]
        counts[start_idx:end_idx] += 1

    # Average overlapping predictions (for most frames, counts=1 since no overlap)
    # But for completeness:
    action_preds = (action_preds / np.maximum(counts, 1)).astype(np.int64)
    agent_preds = (agent_preds / np.maximum(counts, 1)).astype(np.int64)
    target_preds = (target_preds / np.maximum(counts, 1)).astype(np.int64)

    return action_preds, agent_preds, target_preds


def generate_submission(
    model,
    data_dir,
    device,
    output_path='submission_v8.csv',
    min_duration=5,
    sequence_length=100
):
    """
    Generate Kaggle submission file

    Args:
        model: Trained V8BehaviorDetector
        data_dir: Path to test data
        device: torch device
        output_path: Output CSV path
        min_duration: Minimum interval duration
        sequence_length: Sliding window size
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

        # Load and process video (similar to V8Dataset)
        try:
            tracking_df = pd.read_parquet(tracking_file)

            # Standard bodyparts
            standard_bodyparts = [
                'nose', 'ear_left', 'ear_right', 'neck',
                'hip_left', 'hip_right', 'tail_base'
            ]

            tracking_df = tracking_df[tracking_df['bodypart'].isin(standard_bodyparts)]

            if len(tracking_df) == 0 or tracking_df['video_frame'].isna().all():
                continue

            max_frame = tracking_df['video_frame'].max()
            if pd.isna(max_frame):
                continue

            num_frames = int(max_frame) + 1
            num_mice = 4
            num_bodyparts = len(standard_bodyparts)

            # Pivot and fill
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

            # Predict
            action_preds, agent_preds, target_preds = predict_video(
                model, keypoints, device, sequence_length
            )

            # Convert to intervals
            intervals = predictions_to_intervals(
                action_preds=action_preds,
                agent_preds=agent_preds,
                target_preds=target_preds,
                min_duration=min_duration
            )

            # Add to submission
            for interval in intervals:
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
    print(f"\n[OK] Submission saved to {output_path}")
    print(f"  Total intervals: {len(submission)}")

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
    parser = argparse.ArgumentParser(description='V8 Inference for Kaggle Submission')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/v8_5090/best_model.pth')
    parser.add_argument('--config', type=str, default='configs/config_v8_5090.yaml')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output', type=str, default='submission_v8.csv')
    parser.add_argument('--min_duration', type=int, default=5)
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

    # Generate submission
    submission = generate_submission(
        model=model,
        data_dir=data_dir,
        device=device,
        output_path=args.output,
        min_duration=args.min_duration,
        sequence_length=config['sequence_length']
    )

    print(f"\n[OK] Inference complete!")
    print(f"  Submission: {args.output}")
    print(f"  Total predictions: {len(submission)}")


if __name__ == '__main__':
    main()
