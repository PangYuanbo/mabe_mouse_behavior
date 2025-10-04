"""
V9 Dataset - Extends V8 dataset with boundary labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from pathlib import Path

# Import V8 dataset
sys.path.insert(0, str(Path(__file__).parent.parent))
from v8_fine_grained.v8_dataset import V8Dataset
from v9_interval_assembler.boundary_labels import generate_soft_boundary_labels
from v9_interval_assembler.pair_mapping import get_channel_index


class V9Dataset(Dataset):
    """
    V9 Dataset for training the interval assembler

    Wraps V8Dataset and adds boundary label generation

    Args:
        data_dir: Path to kaggle data directory
        split: 'train' or 'val'
        sequence_length: Sequence length (default 100)
        stride: Sliding window stride (default 25)
        sigma: Gaussian smoothing sigma for boundary labels (default 2.0)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 100,
        stride: int = 25,
        sigma: float = 2.0,
        num_actions: int = 28
    ):
        # Load V8 dataset (provides keypoints + frame-level labels)
        self.v8_dataset = V8Dataset(
            data_dir=data_dir,
            split=split,
            sequence_length=sequence_length,
            stride=stride
        )

        self.sequence_length = sequence_length
        self.sigma = sigma
        self.num_actions = num_actions

        print(f"[V9 Dataset] Loaded {len(self.v8_dataset)} sequences from {split} split")

    def __len__(self):
        return len(self.v8_dataset)

    def __getitem__(self, idx):
        """
        Returns:
            keypoints: [T, D] Input keypoint features
            action: [T] Frame-level action labels (from V8)
            agent: [T] Frame-level agent labels
            target: [T] Frame-level target labels
            start_labels: [T, 336] Soft start boundary labels
            end_labels: [T, 336] Soft end boundary labels
            segment_mask: [T, 336] Active segment mask
        """
        # Get V8 sequence
        sequence = self.v8_dataset.sequences[idx]

        keypoints = torch.from_numpy(sequence['keypoints']).float()
        action = torch.from_numpy(sequence['action']).long()
        agent = torch.from_numpy(sequence['agent']).long()
        target = torch.from_numpy(sequence['target']).long()

        T = len(action)

        # Convert frame-level labels to interval list for boundary generation
        intervals = self._frames_to_intervals(action.numpy(), agent.numpy(), target.numpy())

        # Generate soft boundary labels
        start_labels, end_labels, segment_mask = generate_soft_boundary_labels(
            intervals,
            sequence_length=T,
            num_actions=self.num_actions,
            sigma=self.sigma
        )

        return {
            'keypoints': keypoints,
            'action': action,
            'agent': agent,
            'target': target,
            'start_labels': torch.from_numpy(start_labels).float(),
            'end_labels': torch.from_numpy(end_labels).float(),
            'segment_mask': torch.from_numpy(segment_mask).float()
        }

    def _frames_to_intervals(self, action, agent, target):
        """
        Convert frame-level labels to interval list

        KEY FIX: Split only on ACTION changes, not agent/target changes.
        This gives V9 more intervals to learn from.

        Args:
            action: [T] action IDs
            agent: [T] agent IDs
            target: [T] target IDs

        Returns:
            intervals: List of interval dicts
        """
        intervals = []
        T = len(action)

        if T == 0:
            return intervals

        # Track current action segment
        curr_action = action[0]
        start_frame = 0

        for t in range(1, T + 1):
            # Split only when action changes (not agent/target)
            if t == T or action[t] != curr_action:
                end_frame = t - 1

                # For non-background actions, find dominant agent/target in this segment
                if curr_action != 0:
                    # Get agent/target for this segment
                    seg_agents = agent[start_frame:end_frame+1]
                    seg_targets = target[start_frame:end_frame+1]

                    # Use most frequent (mode) agent/target
                    unique_agents, counts_agents = np.unique(seg_agents, return_counts=True)
                    unique_targets, counts_targets = np.unique(seg_targets, return_counts=True)

                    dominant_agent = int(unique_agents[np.argmax(counts_agents)])
                    dominant_target = int(unique_targets[np.argmax(counts_targets)])

                    # Only keep if agent != target
                    if dominant_agent != dominant_target:
                        intervals.append({
                            'action_id': int(curr_action),
                            'agent_id': dominant_agent,
                            'target_id': dominant_target,
                            'start_frame': int(start_frame),
                            'stop_frame': int(end_frame)
                        })

                # Start new segment
                if t < T:
                    curr_action = action[t]
                    start_frame = t

        return intervals


def create_v9_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    sequence_length: int = 100,
    stride: int = 25,
    sigma: float = 2.0,
    num_workers: int = 0,
    pin_memory: bool = True
):
    """
    Create V9 train and validation dataloaders

    Args:
        data_dir: Path to kaggle data
        batch_size: Batch size
        sequence_length: Sequence length
        stride: Sliding window stride
        sigma: Boundary label smoothing
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    train_dataset = V9Dataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        stride=stride,
        sigma=sigma
    )

    val_dataset = V9Dataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        stride=100,  # No overlap for validation
        sigma=sigma
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# Testing
if __name__ == '__main__':
    print("Testing V9 Dataset...")

    DATA_DIR = "C:/Users/aaron/PycharmProjects/mabe_mouse_behavior/data/kaggle"

    print("\n[Test 1] Creating V9 dataset...")
    dataset = V9Dataset(
        data_dir=DATA_DIR,
        split='val',
        sequence_length=100,
        stride=100,
        sigma=2.0
    )

    print(f"  Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        print("\n[Test 2] Getting sample...")
        sample = dataset[0]

        print(f"  Keypoints shape: {sample['keypoints'].shape}")
        print(f"  Action shape: {sample['action'].shape}")
        print(f"  Start labels shape: {sample['start_labels'].shape}")
        print(f"  End labels shape: {sample['end_labels'].shape}")
        print(f"  Segment mask shape: {sample['segment_mask'].shape}")

        print(f"\n  Start labels non-zero: {(sample['start_labels'] > 0).sum().item()}")
        print(f"  End labels non-zero: {(sample['end_labels'] > 0).sum().item()}")

    print("\n[Test 3] Creating dataloaders...")
    try:
        train_loader, val_loader = create_v9_dataloaders(
            data_dir=DATA_DIR,
            batch_size=4,
            num_workers=0
        )

        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

        # Test batch
        batch = next(iter(val_loader))
        print(f"\n  Batch keypoints: {batch['keypoints'].shape}")
        print(f"  Batch start labels: {batch['start_labels'].shape}")

        print("\n[OK] All tests passed!")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("  (This is expected if data directory doesn't exist)")
