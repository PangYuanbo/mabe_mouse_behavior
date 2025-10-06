"""
Test V8 Advanced Post-processing
Compare basic vs advanced inference methods
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Adjust path to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from versions.v8_fine_grained.v8_model import V8BehaviorDetector
from versions.v8_fine_grained.v8_dataset import V8Dataset
from versions.v8_fine_grained.action_mapping import NUM_ACTIONS, ID_TO_ACTION
from versions.v8_fine_grained.submission_utils import evaluate_intervals, predictions_to_intervals
from versions.v8_fine_grained.advanced_postprocessing import (
    sliding_window_inference,
    temporal_smoothing,
    probs_to_intervals_advanced
)
from src.utils.detailed_metrics import compute_interval_per_class_f1, print_detailed_metrics

# Config
DATA_DIR = "C:/Users/aaron/PycharmProjects/mabe_mouse_behavior/data/kaggle"
MODEL_PATH = "checkpoints/v8_5090/best_model.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*60)
print("V8 Post-processing Comparison")
print("="*60)

# Load model (match config_v8_5090.yaml architecture)
print("\nLoading V8 model...")
model = V8BehaviorDetector(
    input_dim=112,
    num_actions=NUM_ACTIONS,
    num_mice=4,
    conv_channels=[128, 256, 512],  # Original architecture
    lstm_hidden=256,
    lstm_layers=2,
    dropout=0.3
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"[OK] Model loaded from {MODEL_PATH}")

# Load validation dataset
print("\nLoading validation data...")
val_dataset = V8Dataset(
    data_dir=DATA_DIR,
    split='val',
    sequence_length=100,
    stride=100  # No overlap for evaluation
)
print(f"[OK] Loaded {len(val_dataset)} validation sequences")

# Test on a few videos
print("\nTesting on validation videos...")
num_test_videos = 3

# Get unique video names
video_names = val_dataset.metadata['video_id'].unique()[:num_test_videos]

for video_name in video_names:
    print(f"\n{'='*60}")
    print(f"Video: {video_name}")
    print(f"{'='*60}")

    # Get all sequences for this video
    video_indices = val_dataset.metadata[val_dataset.metadata['video_id'] == video_name].index.tolist()

    # Reconstruct full video keypoints and labels
    all_keypoints = []
    all_actions = []
    all_agents = []
    all_targets = []

    for idx in video_indices:
        seq = val_dataset.sequences[idx]
        all_keypoints.append(seq['keypoints'])
        all_actions.append(seq['action'])
        all_agents.append(seq['agent'])
        all_targets.append(seq['target'])

    # Concatenate
    full_keypoints = np.concatenate(all_keypoints, axis=0)  # [T, D]
    full_actions = np.concatenate(all_actions, axis=0)  # [T]
    full_agents = np.concatenate(all_agents, axis=0)
    full_targets = np.concatenate(all_targets, axis=0)

    print(f"  Video length: {len(full_keypoints)} frames")

    # Method 1: Basic inference (no overlap, argmax)
    print("\n[1] Basic Inference (no overlap, argmax)...")
    basic_action_preds = []
    basic_agent_preds = []
    basic_target_preds = []

    with torch.no_grad():
        for i in range(0, len(full_keypoints), 100):
            window = full_keypoints[i:i+100]
            if len(window) < 100:
                window = np.pad(window, ((0, 100-len(window)), (0, 0)), mode='edge')

            window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(DEVICE)
            action_logits, agent_logits, target_logits = model(window_tensor)

            action_pred = action_logits.argmax(dim=-1)[0].cpu().numpy()
            agent_pred = agent_logits.argmax(dim=-1)[0].cpu().numpy()
            target_pred = target_logits.argmax(dim=-1)[0].cpu().numpy()

            valid_len = min(100, len(full_keypoints) - i)
            basic_action_preds.append(action_pred[:valid_len])
            basic_agent_preds.append(agent_pred[:valid_len])
            basic_target_preds.append(target_pred[:valid_len])

    basic_action_preds = np.concatenate(basic_action_preds)
    basic_agent_preds = np.concatenate(basic_agent_preds)
    basic_target_preds = np.concatenate(basic_target_preds)

    # Convert to intervals
    basic_intervals = predictions_to_intervals(
        action_preds=basic_action_preds,
        agent_preds=basic_agent_preds,
        target_preds=basic_target_preds,
        min_duration=5
    )

    # Method 2: Advanced inference (sliding window + smoothing + advanced postprocessing)
    print("[2] Advanced Inference (sliding + smoothing + advanced postproc)...")

    # Sliding window inference
    action_probs, agent_probs, target_probs = sliding_window_inference(
        model=model,
        keypoints=full_keypoints,
        sequence_length=100,
        stride=25,  # 75% overlap
        device=DEVICE
    )

    # Temporal smoothing
    action_probs_smooth = temporal_smoothing(action_probs, kernel_size=5, method='median')
    agent_probs_smooth = temporal_smoothing(agent_probs, kernel_size=5, method='median')
    target_probs_smooth = temporal_smoothing(target_probs, kernel_size=5, method='median')

    # Advanced postprocessing
    advanced_intervals = probs_to_intervals_advanced(
        action_probs=action_probs_smooth,
        agent_probs=agent_probs_smooth,
        target_probs=target_probs_smooth,
        action_names=ID_TO_ACTION,
        keypoints=full_keypoints
    )

    # Ground truth intervals
    gt_intervals = predictions_to_intervals(
        action_preds=full_actions,
        agent_preds=full_agents,
        target_preds=full_targets,
        min_duration=1
    )

    # Evaluate
    print("\n" + "-"*60)
    print("Results:")
    print("-"*60)

    # Basic method
    basic_metrics = evaluate_intervals(basic_intervals, gt_intervals, iou_threshold=0.5)
    print(f"Basic Method:")
    print(f"  Intervals: {len(basic_intervals)}")
    print(f"  Precision: {basic_metrics['precision']:.4f}")
    print(f"  Recall:    {basic_metrics['recall']:.4f}")
    print(f"  F1:        {basic_metrics['f1']:.4f}")

    # Advanced method
    advanced_metrics = evaluate_intervals(advanced_intervals, gt_intervals, iou_threshold=0.5)
    print(f"\nAdvanced Method:")
    print(f"  Intervals: {len(advanced_intervals)}")
    print(f"  Precision: {advanced_metrics['precision']:.4f}")
    print(f"  Recall:    {advanced_metrics['recall']:.4f}")
    print(f"  F1:        {advanced_metrics['f1']:.4f}")

    # Improvement
    f1_improvement = advanced_metrics['f1'] - basic_metrics['f1']
    print(f"\nImprovement:")
    print(f"  ΔF1: {f1_improvement:+.4f} ({f1_improvement/max(basic_metrics['f1'], 0.001)*100:+.1f}%)")

print("\n" + "="*60)
print("[OK] Post-processing comparison complete!")
print("="*60)
