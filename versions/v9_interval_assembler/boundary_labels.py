"""
Boundary Label Generation for V9
Generates soft boundary heatmaps with Gaussian smoothing
"""

import numpy as np
from typing import List, Dict, Tuple
from .pair_mapping import get_channel_index


def gaussian_kernel(x: np.ndarray, center: float, sigma: float = 2.0) -> np.ndarray:
    """
    Generate 1D Gaussian kernel

    Args:
        x: Array of positions
        center: Peak position
        sigma: Standard deviation

    Returns:
        Gaussian values at positions x
    """
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def generate_soft_boundary_labels(
    gt_intervals: List[Dict],
    sequence_length: int,
    num_actions: int = 28,
    sigma: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate soft boundary labels with Gaussian smoothing

    Args:
        gt_intervals: List of ground truth intervals
            Each interval: {'action_id', 'agent_id', 'target_id', 'start_frame', 'stop_frame'}
        sequence_length: Number of frames in sequence
        num_actions: Number of action classes (default 28)
        sigma: Gaussian kernel width (default 2.0)

    Returns:
        start_labels: [T, 28×12] Start boundary heatmap
        end_labels: [T, 28×12] End boundary heatmap
        segment_mask: [T, 28×12] Binary mask indicating active segments
    """
    num_channels = num_actions * 12  # 28 actions × 12 pairs

    start_labels = np.zeros((sequence_length, num_channels), dtype=np.float32)
    end_labels = np.zeros((sequence_length, num_channels), dtype=np.float32)
    segment_mask = np.zeros((sequence_length, num_channels), dtype=np.float32)

    # Frame indices for Gaussian kernel
    frames = np.arange(sequence_length)

    for interval in gt_intervals:
        action_id = interval['action_id']
        agent_id = interval['agent_id']
        target_id = interval['target_id']
        start_frame = interval['start_frame']
        stop_frame = interval['stop_frame']

        # Skip background
        if action_id == 0:
            continue

        # Skip invalid pairs
        if agent_id == target_id:
            continue

        # Get channel index for this (action, agent, target) triplet
        try:
            channel = get_channel_index(action_id, agent_id, target_id, num_actions)
        except ValueError:
            # Invalid pair, skip
            continue

        # Generate soft boundary labels with Gaussian smoothing
        start_gauss = gaussian_kernel(frames, start_frame, sigma)
        end_gauss = gaussian_kernel(frames, stop_frame, sigma)

        # Add to existing labels (accumulate for overlapping intervals)
        start_labels[:, channel] = np.maximum(start_labels[:, channel], start_gauss)
        end_labels[:, channel] = np.maximum(end_labels[:, channel], end_gauss)

        # Mark segment region as active
        segment_mask[start_frame:stop_frame+1, channel] = 1.0

    return start_labels, end_labels, segment_mask


def generate_hard_boundary_labels(
    gt_intervals: List[Dict],
    sequence_length: int,
    num_actions: int = 28
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate hard boundary labels (for debugging/ablation)

    Args:
        gt_intervals: List of ground truth intervals
        sequence_length: Number of frames in sequence
        num_actions: Number of action classes

    Returns:
        start_labels: [T, 28×12] Binary start labels
        end_labels: [T, 28×12] Binary end labels
    """
    num_channels = num_actions * 12

    start_labels = np.zeros((sequence_length, num_channels), dtype=np.float32)
    end_labels = np.zeros((sequence_length, num_channels), dtype=np.float32)

    for interval in gt_intervals:
        action_id = interval['action_id']
        agent_id = interval['agent_id']
        target_id = interval['target_id']
        start_frame = interval['start_frame']
        stop_frame = interval['stop_frame']

        # Skip background and invalid pairs
        if action_id == 0 or agent_id == target_id:
            continue

        try:
            channel = get_channel_index(action_id, agent_id, target_id, num_actions)
        except ValueError:
            continue

        # Hard binary labels
        start_labels[start_frame, channel] = 1.0
        end_labels[stop_frame, channel] = 1.0

    return start_labels, end_labels


def visualize_boundary_labels(
    start_labels: np.ndarray,
    end_labels: np.ndarray,
    action_id: int = 1,
    pair_id: int = 0,
    save_path: str = None
):
    """
    Visualize boundary labels for debugging

    Args:
        start_labels: [T, C] Start boundary heatmap
        end_labels: [T, C] End boundary heatmap
        action_id: Action to visualize
        pair_id: Pair to visualize
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt

    channel = action_id * 12 + pair_id
    T = len(start_labels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Start boundary
    axes[0].plot(start_labels[:, channel], label='Start', color='green')
    axes[0].fill_between(range(T), 0, start_labels[:, channel], alpha=0.3, color='green')
    axes[0].set_ylabel('Start Probability')
    axes[0].set_title(f'Boundary Labels - Action {action_id}, Pair {pair_id}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # End boundary
    axes[1].plot(end_labels[:, channel], label='End', color='red')
    axes[1].fill_between(range(T), 0, end_labels[:, channel], alpha=0.3, color='red')
    axes[1].set_ylabel('End Probability')
    axes[1].set_xlabel('Frame')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


# Unit tests
if __name__ == '__main__':
    print("Testing Boundary Label Generation...")

    # Test soft boundary labels
    print("\n[Test 1] Soft boundary labels:")
    gt_intervals = [
        {'action_id': 5, 'agent_id': 0, 'target_id': 1, 'start_frame': 10, 'stop_frame': 20},
        {'action_id': 5, 'agent_id': 1, 'target_id': 0, 'start_frame': 30, 'stop_frame': 45},
        {'action_id': 10, 'agent_id': 2, 'target_id': 3, 'start_frame': 50, 'stop_frame': 60},
    ]

    start_labels, end_labels, seg_mask = generate_soft_boundary_labels(
        gt_intervals, sequence_length=100, num_actions=28, sigma=2.0
    )

    print(f"  Start labels shape: {start_labels.shape}")
    print(f"  End labels shape: {end_labels.shape}")
    print(f"  Segment mask shape: {seg_mask.shape}")
    print(f"  Start labels max: {start_labels.max():.4f}")
    print(f"  End labels max: {end_labels.max():.4f}")

    # Check Gaussian smoothing
    channel_0 = get_channel_index(5, 0, 1, num_actions=28)
    print(f"\n  Channel for (action=5, agent=0, target=1): {channel_0}")
    print(f"  Start peak at frame 10: {start_labels[10, channel_0]:.4f}")
    print(f"  Start at frame 9: {start_labels[9, channel_0]:.4f}")
    print(f"  Start at frame 11: {start_labels[11, channel_0]:.4f}")

    # Test hard boundary labels
    print("\n[Test 2] Hard boundary labels:")
    start_hard, end_hard = generate_hard_boundary_labels(
        gt_intervals, sequence_length=100, num_actions=28
    )
    print(f"  Hard start labels non-zero frames: {np.sum(start_hard > 0)}")
    print(f"  Hard end labels non-zero frames: {np.sum(end_hard > 0)}")

    # Test edge cases
    print("\n[Test 3] Edge cases:")
    # Empty intervals
    start_empty, end_empty, mask_empty = generate_soft_boundary_labels(
        [], sequence_length=100, num_actions=28
    )
    print(f"  Empty intervals - Start sum: {start_empty.sum():.4f}")
    print(f"  Empty intervals - End sum: {end_empty.sum():.4f}")

    # Background action (should be skipped)
    bg_intervals = [
        {'action_id': 0, 'agent_id': 0, 'target_id': 1, 'start_frame': 10, 'stop_frame': 20},
    ]
    start_bg, end_bg, mask_bg = generate_soft_boundary_labels(
        bg_intervals, sequence_length=100, num_actions=28
    )
    print(f"  Background action - Start sum: {start_bg.sum():.4f}")
    print(f"  Background action - End sum: {end_bg.sum():.4f}")

    # Invalid pair (agent == target, should be skipped)
    invalid_intervals = [
        {'action_id': 5, 'agent_id': 2, 'target_id': 2, 'start_frame': 10, 'stop_frame': 20},
    ]
    start_inv, end_inv, mask_inv = generate_soft_boundary_labels(
        invalid_intervals, sequence_length=100, num_actions=28
    )
    print(f"  Invalid pair - Start sum: {start_inv.sum():.4f}")
    print(f"  Invalid pair - End sum: {end_inv.sum():.4f}")

    print("\n[OK] All tests passed!")

    # Optional: visualize (requires matplotlib)
    try:
        print("\n[Visualization] Generating sample plot...")
        visualize_boundary_labels(
            start_labels, end_labels,
            action_id=5, pair_id=0,
            save_path='boundary_labels_sample.png'
        )
    except ImportError:
        print("  Skipping visualization (matplotlib not available)")
