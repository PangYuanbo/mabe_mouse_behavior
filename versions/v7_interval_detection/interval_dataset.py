"""
V7: Interval Detection Dataset with Motion Features
Outputs behavior intervals instead of frame-level labels
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IntervalDetectionDataset(Dataset):
    """
    Dataset for temporal action detection
    Returns: (sequence, intervals)
    - sequence: [seq_len, features]
    - intervals: list of (start_frame, end_frame, action_id, agent_id, target_id)

    Features include:
    - Original coordinates: num_keypoints * 2
    - Speed (motion magnitude): num_keypoints
    - Acceleration: num_keypoints
    Total: num_keypoints * 4 dimensions
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 1000,
        fps: float = 33.3,
        use_motion_features: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.fps = fps
        self.use_motion_features = use_motion_features

        # Load metadata - use MABe22 only (consistent 72 keypoints)
        csv_file = self.data_dir / f'{split}_mabe22.csv'
        if not csv_file.exists():
            csv_file = self.data_dir / f'{split}.csv'
            print(f"Warning: {split}_mabe22.csv not found, using {split}.csv (may have dimension mismatches)")
        self.csv = pd.read_csv(csv_file)

        # Action mapping - all 37 behaviors in MABe dataset
        # 0 = background (no behavior), actions are 1-37
        self.action_to_id = {
            'allogroom': 1, 'approach': 2, 'attack': 3, 'attemptmount': 4,
            'avoid': 5, 'biteobject': 6, 'chase': 7, 'chaseattack': 8,
            'climb': 9, 'defend': 10, 'dig': 11, 'disengage': 12,
            'dominance': 13, 'dominancegroom': 14, 'dominancemount': 15,
            'ejaculate': 16, 'escape': 17, 'exploreobject': 18, 'flinch': 19,
            'follow': 20, 'freeze': 21, 'genitalgroom': 22, 'huddle': 23,
            'intromit': 24, 'mount': 25, 'rear': 26, 'reciprocalsniff': 27,
            'rest': 28, 'run': 29, 'selfgroom': 30, 'shepherd': 31,
            'sniff': 32, 'sniffbody': 33, 'sniffface': 34, 'sniffgenital': 35,
            'submit': 36, 'tussle': 37,
        }
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}

        motion_info = " (with motion features)" if use_motion_features else ""
        print(f"Loaded {len(self.csv)} videos for {split}{motion_info}")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, List[Dict], torch.Tensor]:
        row = self.csv.iloc[idx]
        video_id = row['video_id']
        lab_id = row['lab_id']

        # Load tracking data (use train_tracking for both train and val splits)
        tracking_split = 'train' if self.split in ['train', 'val'] else self.split
        tracking_file = self.data_dir / f'{tracking_split}_tracking' / lab_id / f'{video_id}.parquet'
        tracking_df = pd.read_parquet(tracking_file)

        # Convert to wide format
        tracking_pivot = tracking_df.pivot_table(
            index='video_frame',
            columns=['mouse_id', 'bodypart'],
            values=['x', 'y'],
            aggfunc='first'
        )
        tracking_pivot.columns = ['_'.join(map(str, col)).strip()
                                  for col in tracking_pivot.columns.values]
        tracking_pivot = tracking_pivot.sort_index()

        keypoints = tracking_pivot.values.astype(np.float32)
        keypoints = np.nan_to_num(keypoints, nan=0.0)

        # Add motion features if enabled
        if self.use_motion_features:
            keypoints = self._add_motion_features(keypoints)

        # Load annotations (intervals) - use train_annotation for both train and val
        anno_split = 'train' if self.split in ['train', 'val'] else self.split
        annotation_file = self.data_dir / f'{anno_split}_annotation' / lab_id / f'{video_id}.parquet'
        intervals = []

        if annotation_file.exists():
            anno_df = pd.read_parquet(annotation_file)

            for _, anno_row in anno_df.iterrows():
                intervals.append({
                    'start_frame': int(anno_row['start_frame']),
                    'end_frame': int(anno_row['stop_frame']),
                    'action_id': self.action_to_id[anno_row['action']],
                    'agent_id': int(anno_row['agent_id']),
                    'target_id': int(anno_row['target_id']),
                })

        # Create frame-level labels from intervals (for dual metric evaluation)
        num_frames = len(keypoints)
        frame_labels = np.zeros(num_frames, dtype=np.int64)  # 0 = background

        for interval in intervals:
            start = interval['start_frame']
            end = interval['end_frame']
            action_id = interval['action_id']
            # Mark frames in this interval with the action_id
            for frame in range(start, min(end + 1, num_frames)):
                # If multiple overlapping behaviors, keep the non-background one
                if frame_labels[frame] == 0:  # Only assign if background
                    frame_labels[frame] = action_id
                elif action_id > 0:  # If both are behaviors, keep the higher priority
                    frame_labels[frame] = max(frame_labels[frame], action_id)

        # Pad or truncate to sequence_length
        if num_frames < self.sequence_length:
            padding = np.zeros((self.sequence_length - num_frames, keypoints.shape[1]),
                             dtype=np.float32)
            keypoints = np.concatenate([keypoints, padding], axis=0)

            # Pad frame labels
            label_padding = np.zeros(self.sequence_length - num_frames, dtype=np.int64)
            frame_labels = np.concatenate([frame_labels, label_padding], axis=0)
        else:
            keypoints = keypoints[:self.sequence_length]
            frame_labels = frame_labels[:self.sequence_length]

            # Adjust intervals to sequence_length
            valid_intervals = []
            for interval in intervals:
                if interval['start_frame'] < self.sequence_length:
                    interval['end_frame'] = min(interval['end_frame'], self.sequence_length - 1)
                    valid_intervals.append(interval)
            intervals = valid_intervals

        return torch.from_numpy(keypoints), intervals, torch.from_numpy(frame_labels)

    def _add_motion_features(self, keypoints):
        """
        Add speed and acceleration features for each keypoint

        Args:
            keypoints: [T, D] array where D = num_keypoints * 2 (x, y coordinates)

        Returns:
            enhanced_keypoints: [T, D*2] array with original + speed + acceleration
            - Original: D dimensions (x, y for each keypoint)
            - Speed: D//2 dimensions (magnitude for each keypoint)
            - Acceleration: D//2 dimensions (magnitude for each keypoint)
            - Total: D + D//2 + D//2 = D*2
        """
        dt = 1.0 / self.fps  # Time interval between frames
        T, D = keypoints.shape
        num_keypoints = D // 2  # Number of (x, y) coordinate pairs

        # Reshape to [T, num_keypoints, 2] for easier computation
        coords = keypoints.reshape(T, num_keypoints, 2)

        # 1. Compute velocity (speed) for each keypoint
        # velocity[t] = (position[t] - position[t-1]) / dt
        velocity = np.zeros_like(coords)
        velocity[1:] = (coords[1:] - coords[:-1]) / dt
        velocity[0] = velocity[1]  # Copy second frame to first (no prior frame)

        # Compute speed magnitude: sqrt(vx^2 + vy^2)
        speed = np.sqrt(np.sum(velocity ** 2, axis=2))  # [T, num_keypoints]

        # 2. Compute acceleration
        # acceleration[t] = (velocity[t] - velocity[t-1]) / dt
        acceleration_vec = np.zeros_like(velocity)
        acceleration_vec[1:] = (velocity[1:] - velocity[:-1]) / dt
        acceleration_vec[0] = acceleration_vec[1]

        # Compute acceleration magnitude
        acceleration = np.sqrt(np.sum(acceleration_vec ** 2, axis=2))  # [T, num_keypoints]

        # 3. Concatenate: [original, speed, acceleration]
        # Original: [T, D]
        # Speed: [T, num_keypoints]
        # Acceleration: [T, num_keypoints]
        enhanced_keypoints = np.concatenate([keypoints, speed, acceleration], axis=1)

        return enhanced_keypoints


def collate_interval_fn(batch):
    """
    Collate function for interval detection
    Returns:
        sequences: [batch_size, seq_len, features]
        intervals: list of list of dicts
        frame_labels: [batch_size, seq_len] (for dual metric evaluation)
    """
    sequences = []
    all_intervals = []
    frame_labels = []

    for seq, intervals, labels in batch:
        sequences.append(seq)
        all_intervals.append(intervals)
        frame_labels.append(labels)

    sequences = torch.stack(sequences)
    frame_labels = torch.stack(frame_labels)

    return sequences, all_intervals, frame_labels
