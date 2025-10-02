"""
V7: Interval Detection Dataset
Outputs behavior intervals instead of frame-level labels
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple


class IntervalDetectionDataset(Dataset):
    """
    Dataset for temporal action detection
    Returns: (sequence, intervals)
    - sequence: [seq_len, features]
    - intervals: list of (start_frame, end_frame, action_id, agent_id, target_id)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 1000,
        fps: float = 33.3,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.fps = fps

        # Load metadata
        self.csv = pd.read_csv(self.data_dir / f'{split}.csv')

        # Action mapping
        self.action_to_id = {
            'attack': 0,
            'avoid': 1,
            'chase': 2,
            'chaseattack': 3,
        }
        self.id_to_action = {v: k for k, v in self.action_to_id.items()}

        print(f"Loaded {len(self.csv)} videos for {split}")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, List[Dict]]:
        row = self.csv.iloc[idx]
        video_id = row['video_id']
        lab_id = row['lab_id']

        # Load tracking data
        tracking_file = self.data_dir / f'{self.split}_tracking' / lab_id / f'{video_id}.parquet'
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

        # Load annotations (intervals)
        annotation_file = self.data_dir / f'{self.split}_annotation' / lab_id / f'{video_id}.parquet'
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

        # Pad or truncate to sequence_length
        num_frames = len(keypoints)
        if num_frames < self.sequence_length:
            padding = np.zeros((self.sequence_length - num_frames, keypoints.shape[1]),
                             dtype=np.float32)
            keypoints = np.concatenate([keypoints, padding], axis=0)
        else:
            keypoints = keypoints[:self.sequence_length]

            # Adjust intervals to sequence_length
            valid_intervals = []
            for interval in intervals:
                if interval['start_frame'] < self.sequence_length:
                    interval['end_frame'] = min(interval['end_frame'], self.sequence_length - 1)
                    valid_intervals.append(interval)
            intervals = valid_intervals

        return torch.from_numpy(keypoints), intervals


def collate_interval_fn(batch):
    """
    Collate function for interval detection
    Returns:
        sequences: [batch_size, seq_len, features]
        intervals: list of list of dicts
    """
    sequences = []
    all_intervals = []

    for seq, intervals in batch:
        sequences.append(seq)
        all_intervals.append(intervals)

    sequences = torch.stack(sequences)

    return sequences, all_intervals
