"""
Dataset loader for real Kaggle MABe competition data
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class KaggleMABeDataset(Dataset):
    """
    Dataset for Kaggle MABe competition

    Data structure:
    - train.csv: metadata with sequence_id, dataset columns
    - train_tracking/{dataset}/{sequence_id}.parquet: keypoint coordinates
    - train_annotation/{dataset}/{sequence_id}.parquet: behavior labels
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 100,
        stride: int = 25,
        max_sequences: Optional[int] = None,
        use_feature_engineering: bool = True,
        feature_engineer=None,
    ):
        """
        Args:
            data_dir: Path to kaggle data directory
            split: 'train' or 'val' (will split from train data)
            sequence_length: Length of each sequence
            stride: Stride for sliding window
            max_sequences: Maximum number of sequences to load (for debugging)
            use_feature_engineering: Whether to apply feature engineering
            feature_engineer: Feature engineering instance
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.use_feature_engineering = use_feature_engineering
        self.feature_engineer = feature_engineer

        # Load metadata
        print(f"Loading {split} data from {data_dir}...")
        train_csv = pd.read_csv(self.data_dir / "train.csv")

        # Split train/val (80/20)
        np.random.seed(42)
        video_ids = train_csv['video_id'].unique()
        np.random.shuffle(video_ids)

        split_idx = int(len(video_ids) * 0.8)
        if split == 'train':
            selected_ids = video_ids[:split_idx]
        else:
            selected_ids = video_ids[split_idx:]

        self.metadata = train_csv[train_csv['video_id'].isin(selected_ids)].copy()

        # Filter to only labs with annotation files (exclude test sets)
        # MABe22_keypoints and MABe22_movies don't have annotations
        labs_with_annotations = [
            'AdaptableSnail', 'BoisterousParrot', 'CRIM13', 'CalMS21_supplemental',
            'CalMS21_task1', 'CalMS21_task2', 'CautiousGiraffe', 'DeliriousFly',
            'ElegantMink', 'GroovyShrew', 'InvincibleJellyfish', 'JovialSwallow',
            'LyricalHare', 'NiftyGoldfinch', 'PleasantMeerkat', 'ReflectiveManatee',
            'SparklingTapir', 'TranquilPanther', 'UppityFerret'
        ]
        self.metadata = self.metadata[self.metadata['lab_id'].isin(labs_with_annotations)].copy()

        print(f"Total videos in metadata: {len(self.metadata)}")

        # Load all sequences and create sliding windows
        self.sequences = []
        self._load_sequences(max_sequences=max_sequences)

        print(f"✓ Loaded {len(self.sequences)} sequences for {split}")

        # Determine max input dimension across all sequences
        if len(self.sequences) > 0:
            self.max_input_dim = max(seq['keypoints'].shape[1] for seq in self.sequences)
            print(f"✓ Max input dimension: {self.max_input_dim}")
        else:
            self.max_input_dim = 0

    def _load_sequences(self, max_sequences=None):
        """Load all sequences and create sliding windows"""
        loaded_videos = 0

        for idx, row in self.metadata.iterrows():
            video_id = row['video_id']
            lab_id = row['lab_id']

            # Load tracking data (keypoints)
            tracking_file = self.data_dir / "train_tracking" / lab_id / f"{video_id}.parquet"
            if not tracking_file.exists():
                print(f"Warning: Tracking file not found for {lab_id}/{video_id}")
                continue

            try:
                tracking_df = pd.read_parquet(tracking_file)

                # Load annotation data (labels)
                annotation_file = self.data_dir / "train_annotation" / lab_id / f"{video_id}.parquet"
                if not annotation_file.exists():
                    print(f"Warning: Annotation file not found for {lab_id}/{video_id}")
                    continue

                annotation_df = pd.read_parquet(annotation_file)

                # Extract keypoints and labels
                keypoints, labels = self._process_sequence(tracking_df, annotation_df)

                if keypoints is None:
                    print(f"Warning: Could not process keypoints/labels for {lab_id}/{video_id}")
                    continue

                if len(keypoints) < self.sequence_length:
                    print(f"Warning: Sequence too short for {lab_id}/{video_id}: {len(keypoints)} < {self.sequence_length}")
                    continue

                # Create sliding windows
                num_windows_before = len(self.sequences)
                for start_idx in range(0, len(keypoints) - self.sequence_length + 1, self.stride):
                    end_idx = start_idx + self.sequence_length

                    window_keypoints = keypoints[start_idx:end_idx]
                    window_labels = labels[start_idx:end_idx]

                    self.sequences.append({
                        'keypoints': window_keypoints,
                        'labels': window_labels,
                        'video_id': video_id,
                        'lab_id': lab_id,
                    })

                num_windows = len(self.sequences) - num_windows_before
                print(f"  Loaded {lab_id}/{video_id}: {num_windows} windows from {len(keypoints)} frames")

                loaded_videos += 1

                # Stop if we've reached max_sequences
                if max_sequences and loaded_videos >= max_sequences:
                    print(f"Reached max_sequences limit: {max_sequences}")
                    break

            except Exception as e:
                print(f"Warning: Failed to load {lab_id}/{video_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    def _process_sequence(self, tracking_df, annotation_df) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process tracking and annotation data

        Tracking format: video_frame, mouse_id, bodypart, x, y (long format)
        Annotation format: agent_id, target_id, action, start_frame, stop_frame (event-based)

        Returns:
            keypoints: [T, num_keypoints * 2] array
            labels: [T] array of behavior class indices
        """
        try:
            # 1. Convert tracking from long to wide format
            # Pivot to get one row per frame with all keypoints
            tracking_pivot = tracking_df.pivot_table(
                index='video_frame',
                columns=['mouse_id', 'bodypart'],
                values=['x', 'y'],
                aggfunc='first'  # In case of duplicates
            )

            # Flatten multi-index columns
            tracking_pivot.columns = ['_'.join(map(str, col)).strip() for col in tracking_pivot.columns.values]
            tracking_pivot = tracking_pivot.sort_index()

            # Get numpy array
            keypoints = tracking_pivot.values.astype(np.float32)

            # Handle missing values
            keypoints = np.nan_to_num(keypoints, nan=0.0)

            # 2. Convert annotation from event-based to frame-by-frame
            num_frames = len(tracking_pivot)
            labels = np.zeros(num_frames, dtype=np.int64)  # Default: 0 = other/none

            # Map actions to class indices
            # For now, group actions into broad categories
            # 0 = background/other
            # 1 = social investigation (sniff, approach)
            # 2 = mating behaviors (mount, intromit)
            # 3 = aggressive behaviors (attack, chase, bite)
            action_mapping = {
                # Social investigation
                'sniff': 1, 'sniffgenital': 1, 'sniffface': 1, 'sniffbody': 1,
                'reciprocalsniff': 1, 'approach': 1, 'follow': 1,
                # Mating behaviors
                'mount': 2, 'intromit': 2, 'attemptmount': 2, 'ejaculate': 2,
                # Aggressive behaviors
                'attack': 3, 'chase': 3, 'chaseattack': 3, 'bite': 3,
                'dominance': 3, 'defend': 3, 'flinch': 3,
                # Everything else as background
                'rear': 0, 'avoid': 0, 'escape': 0, 'freeze': 0,
                'selfgroom': 0, 'allogroom': 0, 'rest': 0, 'dig': 0,
                'climb': 0, 'shepherd': 0, 'disengage': 0, 'run': 0,
                'exploreobject': 0, 'biteobject': 0, 'dominancegroom': 0,
                'huddle': 0, 'other': 0,
            }

            # Fill in labels for each event
            for _, row in annotation_df.iterrows():
                action = row['action']
                start_frame = row['start_frame']
                stop_frame = row['stop_frame']

                # Map action to label
                label = action_mapping.get(action, 0)

                # Fill frames in this range
                for frame in range(start_frame, stop_frame + 1):
                    if frame < num_frames:
                        # If multiple behaviors overlap, keep the higher priority one
                        if label > labels[frame]:
                            labels[frame] = label

            return keypoints, labels

        except Exception as e:
            print(f"Error in _process_sequence: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            keypoints: [sequence_length, input_dim]
            labels: [sequence_length]
        """
        item = self.sequences[idx]

        keypoints = item['keypoints'].astype(np.float32)
        labels = item['labels'].astype(np.int64)

        # Apply feature engineering if enabled
        if self.use_feature_engineering and self.feature_engineer is not None:
            try:
                keypoints = self.feature_engineer.extract_all_features(
                    keypoints,
                    include_pca=True,
                    include_temporal=True
                )
            except Exception as e:
                # Fallback to original keypoints if feature engineering fails
                pass

        # Pad keypoints to max_input_dim to ensure consistent dimensions
        current_dim = keypoints.shape[1]
        if current_dim < self.max_input_dim:
            padding = np.zeros((keypoints.shape[0], self.max_input_dim - current_dim), dtype=np.float32)
            keypoints = np.concatenate([keypoints, padding], axis=1)

        # Convert to tensors
        keypoints = torch.FloatTensor(keypoints)
        labels = torch.LongTensor(labels)

        return keypoints, labels


def create_kaggle_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    sequence_length: int = 100,
    num_workers: int = 4,
    use_feature_engineering: bool = True,
    max_sequences: Optional[int] = None,
):
    """
    Create train and validation dataloaders for Kaggle data

    Args:
        data_dir: Path to kaggle data directory
        batch_size: Batch size
        sequence_length: Sequence length
        num_workers: Number of data loading workers
        use_feature_engineering: Whether to apply feature engineering
        max_sequences: Maximum sequences to load (for debugging)

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader

    # Feature engineering setup
    feature_engineer = None
    if use_feature_engineering:
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from data.feature_engineering import MouseFeatureEngineer

            feature_engineer = MouseFeatureEngineer(num_mice=2, num_keypoints=7)
        except Exception as e:
            print(f"Warning: Could not load feature engineering: {e}")
            use_feature_engineering = False

    # Create datasets
    train_dataset = KaggleMABeDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        stride=sequence_length // 4,  # 75% overlap
        max_sequences=max_sequences,
        use_feature_engineering=use_feature_engineering,
        feature_engineer=feature_engineer,
    )

    val_dataset = KaggleMABeDataset(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        stride=sequence_length,  # No overlap for validation
        max_sequences=max_sequences // 5 if max_sequences else None,
        use_feature_engineering=use_feature_engineering,
        feature_engineer=feature_engineer,
    )

    # Ensure both datasets use the same max_input_dim
    max_dim = max(train_dataset.max_input_dim, val_dataset.max_input_dim)
    train_dataset.max_input_dim = max_dim
    val_dataset.max_input_dim = max_dim
    print(f"✓ Unified max input dimension: {max_dim}")

    # Fit PCA on training data if using feature engineering
    if use_feature_engineering and feature_engineer is not None:
        print("Fitting PCA on training data...")
        all_features = []
        for i in range(min(100, len(train_dataset))):  # Use first 100 sequences
            item = train_dataset.sequences[i]
            features = feature_engineer.extract_all_features(
                item['keypoints'],
                include_pca=False,
                include_temporal=True
            )
            all_features.append(features)

        all_features = np.vstack(all_features)
        feature_engineer.fit_pca(all_features)
        print(f"✓ PCA fitted with explained variance: {feature_engineer.pca.explained_variance_ratio_.sum():.3f}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
