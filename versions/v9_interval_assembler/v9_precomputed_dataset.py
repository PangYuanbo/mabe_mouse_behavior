"""
V9 Dataset using precomputed V8 features
Much faster than computing V8 features on-the-fly
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from v9_interval_assembler.boundary_labels import generate_soft_boundary_labels


class V9PrecomputedDataset(Dataset):
    """
    V9 Dataset loading precomputed V8 features from HDF5

    Args:
        h5_file: Path to precomputed V8 features (e.g., 'data/v8_features/train_v8_features.h5')
        sigma: Gaussian smoothing sigma for boundary labels
        num_actions: Number of action classes
    """

    def __init__(
        self,
        h5_file: str,
        sigma: float = 2.0,
        num_actions: int = 28
    ):
        self.h5_file = Path(h5_file)
        self.sigma = sigma
        self.num_actions = num_actions

        # Open HDF5 file in read mode
        self.h5 = h5py.File(self.h5_file, 'r')

        # Get datasets
        self.action_logits = self.h5['action_logits']
        self.agent_logits = self.h5['agent_logits']
        self.target_logits = self.h5['target_logits']
        self.action_labels = self.h5['action_labels']
        self.agent_labels = self.h5['agent_labels']
        self.target_labels = self.h5['target_labels']

        # Get metadata
        self.num_sequences = self.h5.attrs['num_sequences']
        self.sequence_length = self.h5.attrs['sequence_length']

        print(f"[V9 Precomputed] Loaded {self.num_sequences} sequences from {h5_file}")

        # DEBUG: Check first sequence intervals
        if self.num_sequences > 0:
            sample = self.__getitem__(0)
            num_intervals = (sample['start_labels'].sum(dim=-1) > 0.1).sum().item()
            print(f"[DEBUG] First sequence has {num_intervals} boundary frames with labels")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Returns:
            v8_logits: [T, 36] Concatenated V8 logits (action + agent + target)
            action: [T] Frame-level action labels
            agent: [T] Frame-level agent labels
            target: [T] Frame-level target labels
            start_labels: [T, 336] Soft start boundary labels
            end_labels: [T, 336] Soft end boundary labels
            segment_mask: [T, 336] Active segment mask
        """
        # Load V8 logits (already computed!)
        action_logits = self.action_logits[idx]  # [T, 28]
        agent_logits = self.agent_logits[idx]    # [T, 4]
        target_logits = self.target_logits[idx]  # [T, 4]

        # Concatenate V8 outputs
        v8_logits = np.concatenate([action_logits, agent_logits, target_logits], axis=-1)
        # v8_logits: [T, 36]

        # Load labels
        action = self.action_labels[idx]  # [T]
        agent = self.agent_labels[idx]
        target = self.target_labels[idx]

        # Convert frame-level labels to intervals
        intervals = self._frames_to_intervals(action, agent, target)

        # Generate soft boundary labels
        T = len(action)
        start_labels, end_labels, segment_mask = generate_soft_boundary_labels(
            intervals,
            sequence_length=T,
            num_actions=self.num_actions,
            sigma=self.sigma
        )

        return {
            'v8_logits': torch.from_numpy(v8_logits).float(),
            'action': torch.from_numpy(action).long(),
            'agent': torch.from_numpy(agent).long(),
            'target': torch.from_numpy(target).long(),
            'start_labels': torch.from_numpy(start_labels).float(),
            'end_labels': torch.from_numpy(end_labels).float(),
            'segment_mask': torch.from_numpy(segment_mask).float()
        }

    def _frames_to_intervals(self, action, agent, target):
        """
        Convert frame-level labels to interval list

        KEY FIX: Split only on ACTION changes, not agent/target changes.
        This gives V9 more intervals to learn from.
        """
        intervals = []
        T = len(action)

        if T == 0:
            return intervals

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

                if t < T:
                    curr_action = action[t]
                    start_frame = t

        return intervals

    def __del__(self):
        """Close HDF5 file on deletion"""
        if hasattr(self, 'h5'):
            self.h5.close()


def create_v9_precomputed_dataloaders(
    train_h5: str,
    val_h5: str,
    batch_size: int = 128,
    sigma: float = 2.0,
    num_workers: int = 0,
    pin_memory: bool = True
):
    """
    Create V9 dataloaders using precomputed V8 features

    Args:
        train_h5: Path to train V8 features
        val_h5: Path to val V8 features
        batch_size: Batch size
        sigma: Boundary label smoothing
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, val_loader
    """
    train_dataset = V9PrecomputedDataset(
        h5_file=train_h5,
        sigma=sigma
    )

    val_dataset = V9PrecomputedDataset(
        h5_file=val_h5,
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
    print("Testing V9 Precomputed Dataset...")

    H5_FILE = "data/v8_features/val_v8_features.h5"

    if not Path(H5_FILE).exists():
        print(f"\n[ERROR] Feature file not found: {H5_FILE}")
        print("Please run precompute_v8_features.py first!")
    else:
        print(f"\n[Test 1] Loading dataset from {H5_FILE}...")
        dataset = V9PrecomputedDataset(H5_FILE, sigma=2.0)

        print(f"  Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            print("\n[Test 2] Getting sample...")
            sample = dataset[0]

            print(f"  V8 logits shape: {sample['v8_logits'].shape}")
            print(f"  Action shape: {sample['action'].shape}")
            print(f"  Start labels shape: {sample['start_labels'].shape}")
            print(f"  End labels shape: {sample['end_labels'].shape}")

            print(f"\n  Start labels non-zero: {(sample['start_labels'] > 0).sum().item()}")
            print(f"  End labels non-zero: {(sample['end_labels'] > 0).sum().item()}")

        print("\n[Test 3] Creating dataloader...")
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        print(f"  Batch v8_logits: {batch['v8_logits'].shape}")
        print(f"  Batch start labels: {batch['start_labels'].shape}")

        print("\n[OK] All tests passed!")
