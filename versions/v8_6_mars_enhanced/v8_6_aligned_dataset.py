"""
V8.6 Aligned Dataset (No Freeze Labels)
========================================
MARS-Enhanced Features with Rare Behavior Oversampling

PRESERVED from V8.6:
- MARSFeatureExtractor (~370 dims)
- Rare behavior oversampling (×3)
- Data augmentation (coordinate noise)

ALIGNED with V8.5:
- Sequence length: 100 frames (not 150)
- Stride: 25 (not 37)
- Returns 3 labels only (action/agent/target, no freeze)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from .feature_engineering import MARSFeatureExtractor
from .action_mapping import ACTION_TO_ID


class V86AlignedDataset(Dataset):
    """
    V8.6 Aligned Dataset with MARS-enhanced features

    Returns (SAME AS V8.5):
        keypoints: [T, ~370] MARS features
        action_labels: [T] action class IDs (0-37)
        agent_labels: [T] agent mouse IDs (0-3)
        target_labels: [T] target mouse IDs (0-3)
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 100,  # SAME AS V8.5 (not 150!)
        stride: int = 25,            # SAME AS V8.5 (not 37!)
        fps: float = 33.3,
        oversample_rare: bool = True,  # Oversample rare behaviors
        augment: bool = True,          # Data augmentation
        coord_noise_std: float = 0.01  # ALIGNED with config (not 2.0!)
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride if split == 'train' else sequence_length
        self.fps = fps
        self.oversample_rare = oversample_rare and (split == 'train')
        self.augment = augment and (split == 'train')
        self.coord_noise_std = coord_noise_std

        # MARS feature extractor
        self.feature_extractor = MARSFeatureExtractor(fps=fps)

        # Rare behaviors to oversample (from V8.5 frequency analysis)
        self.rare_behaviors = [
            'freeze', 'escape', 'dominancegroom', 'attemptmount',
            'ejaculate', 'flinch', 'disengage', 'allogroom',
            'biteobject', 'submit', 'avoid'
        ]
        self.rare_behavior_ids = [ACTION_TO_ID[name] for name in self.rare_behaviors if name in ACTION_TO_ID]

        # Load metadata
        csv_path = self.data_dir / f"{split}.csv" if (self.data_dir / f"{split}.csv").exists() else self.data_dir / "train.csv"
        metadata = pd.read_csv(csv_path)

        # Filter to train/val split
        np.random.seed(42)
        video_ids = metadata['video_id'].unique()
        np.random.shuffle(video_ids)
        split_idx = int(len(video_ids) * 0.8)

        if split == 'train':
            selected_ids = video_ids[:split_idx]
        else:
            selected_ids = video_ids[split_idx:]

        metadata = metadata[metadata['video_id'].isin(selected_ids)]

        # Filter to labs with annotations
        labs_with_annotations = [
            'AdaptableSnail', 'BoisterousParrot', 'CRIM13', 'CalMS21_supplemental',
            'CalMS21_task1', 'CalMS21_task2', 'CautiousGiraffe', 'DeliriousFly',
            'ElegantMink', 'GroovyShrew', 'InvincibleJellyfish', 'JovialSwallow',
            'LyricalHare', 'NiftyGoldfinch', 'PleasantMeerkat', 'ReflectiveManatee',
            'SparklingTapir', 'TranquilPanther', 'UppityFerret'
        ]
        metadata = metadata[metadata['lab_id'].isin(labs_with_annotations)]

        self.metadata = metadata
        self.sequences = []

        print(f"Loading {split} data from {data_dir}...")
        self._load_sequences()

        # Oversample rare behaviors if enabled
        if self.oversample_rare:
            self._oversample_rare_behaviors()

        print(f"[OK] Loaded {len(self.sequences)} sequences for {split}")

    def _load_sequences(self):
        """Load all video sequences and create sliding windows"""
        from tqdm import tqdm

        failed_count = 0

        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc=f"Processing {self.split} videos"):
            video_id = row['video_id']
            lab_id = row['lab_id']

            tracking_file = self.data_dir / "train_tracking" / lab_id / f"{video_id}.parquet"
            annotation_file = self.data_dir / "train_annotation" / lab_id / f"{video_id}.parquet"

            if not tracking_file.exists():
                failed_count += 1
                continue

            # Process video
            keypoints, action_labels, agent_labels, target_labels = self._process_video(
                tracking_file, annotation_file
            )

            if keypoints is None or len(keypoints) < self.sequence_length:
                failed_count += 1
                continue

            # Create sliding windows
            num_frames = len(keypoints)
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length

                # Check if this sequence contains any rare behavior
                seq_actions = action_labels[start_idx:end_idx]
                has_rare = any(action_id in self.rare_behavior_ids for action_id in seq_actions)

                self.sequences.append({
                    'keypoints': keypoints[start_idx:end_idx],
                    'action': action_labels[start_idx:end_idx],
                    'agent': agent_labels[start_idx:end_idx],
                    'target': target_labels[start_idx:end_idx],
                    'has_rare': has_rare  # For oversampling
                })

        if failed_count > 0:
            print(f"  [!] {failed_count} videos failed to load")

    def _process_video(
        self,
        tracking_file: Path,
        annotation_file: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single video with MARS feature extraction

        Returns (NO FREEZE LABELS):
            keypoints: [T, ~370] MARS-enhanced features
            action_labels: [T] action class IDs
            agent_labels: [T] agent mouse IDs (0-3)
            target_labels: [T] target mouse IDs (0-3)
        """
        try:
            # Load tracking data
            tracking_df = pd.read_parquet(tracking_file)

            # Standard bodyparts
            standard_bodyparts = [
                'nose', 'ear_left', 'ear_right', 'neck',
                'hip_left', 'hip_right', 'tail_base'
            ]

            # Filter to standard bodyparts
            tracking_df = tracking_df[tracking_df['bodypart'].isin(standard_bodyparts)]

            if len(tracking_df) == 0 or tracking_df['video_frame'].isna().all():
                return None, None, None, None

            # Create fixed-size keypoint array
            max_frame = tracking_df['video_frame'].max()
            if pd.isna(max_frame):
                return None, None, None, None

            num_frames = int(max_frame) + 1
            num_mice = 4
            num_bodyparts = len(standard_bodyparts)

            # Pivot x and y separately
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

            # Initialize raw keypoints [T, 56]
            keypoints_raw = np.zeros((num_frames, num_mice * num_bodyparts * 2), dtype=np.float32)

            # Fill in data
            for mouse_id in range(1, 5):
                for bp_idx, bodypart in enumerate(standard_bodyparts):
                    if (mouse_id, bodypart) in x_pivot.columns:
                        frames = x_pivot.index.values.astype(int)
                        x_vals = x_pivot[(mouse_id, bodypart)].values
                        y_vals = y_pivot[(mouse_id, bodypart)].values

                        base_idx = (mouse_id - 1) * num_bodyparts * 2 + bp_idx * 2
                        keypoints_raw[frames, base_idx] = x_vals
                        keypoints_raw[frames, base_idx + 1] = y_vals

            keypoints_raw = np.nan_to_num(keypoints_raw, nan=0.0)

            # Initialize labels
            action_labels = np.zeros(num_frames, dtype=np.int64)
            agent_labels = np.zeros(num_frames, dtype=np.int64)
            target_labels = np.zeros(num_frames, dtype=np.int64)

            # Load annotations
            if annotation_file.exists():
                annotation_df = pd.read_parquet(annotation_file)

                for _, row in annotation_df.iterrows():
                    action = row['action']
                    start_frame = row['start_frame']
                    stop_frame = row['stop_frame']
                    agent_id = row.get('agent_id', 1)
                    target_id = row.get('target_id', 2)

                    # Map action to ID
                    action_id = ACTION_TO_ID.get(action, 0)

                    # Map agent/target
                    agent_idx = int(agent_id) - 1 if isinstance(agent_id, (int, float)) else 0
                    target_idx = int(target_id) - 1 if isinstance(target_id, (int, float)) else 1

                    agent_idx = np.clip(agent_idx, 0, 3)
                    target_idx = np.clip(target_idx, 0, 3)

                    # Fill frames
                    for frame in range(start_frame, min(stop_frame + 1, num_frames)):
                        if action_id > action_labels[frame]:
                            action_labels[frame] = action_id
                            agent_labels[frame] = agent_idx
                            target_labels[frame] = target_idx

            # Extract MARS features
            keypoints_enhanced = self.feature_extractor.extract_all_features(keypoints_raw)

            return keypoints_enhanced, action_labels, agent_labels, target_labels

        except Exception as e:
            print(f"Error processing video: {e}")
            return None, None, None, None

    def _oversample_rare_behaviors(self):
        """
        Oversample sequences containing rare behaviors × 3

        Rare behaviors: freeze, escape, dominancegroom, etc.
        """
        rare_sequences = [seq for seq in self.sequences if seq['has_rare']]

        if len(rare_sequences) > 0:
            # Repeat rare sequences 2 more times (total 3x)
            oversampled = rare_sequences * 2
            self.sequences.extend(oversampled)

            print(f"  [Oversample] Added {len(oversampled)} rare behavior sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        keypoints = seq['keypoints'].copy()  # [T, ~370]
        action = seq['action'].copy()
        agent = seq['agent'].copy()
        target = seq['target'].copy()

        # Data augmentation (only during training)
        if self.augment:
            # Coordinate noise: add Gaussian noise to raw coordinates
            # Only apply to first 56 dims (raw coordinates)
            if np.random.rand() < 0.5:
                noise = np.random.randn(keypoints.shape[0], 56) * self.coord_noise_std
                keypoints[:, :56] += noise

        # Convert to tensors
        keypoints_t = torch.FloatTensor(keypoints)
        action_t = torch.LongTensor(action)
        agent_t = torch.LongTensor(agent)
        target_t = torch.LongTensor(target)

        return keypoints_t, action_t, agent_t, target_t


def create_v86_aligned_dataloaders(
    data_dir: str,
    batch_size: int = 256,          # SAME AS V8.5
    sequence_length: int = 100,     # SAME AS V8.5
    stride: int = 25,               # SAME AS V8.5
    num_workers: int = 0,
    pin_memory: bool = True,
    oversample_rare: bool = True,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create V8.6 Aligned train and validation dataloaders

    Returns:
        train_loader, val_loader
    """
    train_dataset = V86AlignedDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        stride=stride,
        oversample_rare=oversample_rare,
        augment=augment
    )

    val_dataset = V86AlignedDataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        stride=sequence_length,  # No overlap for validation
        oversample_rare=False,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader
