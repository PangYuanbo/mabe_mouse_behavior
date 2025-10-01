"""
Advanced dataset with feature engineering for MABe mouse behavior detection
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from .feature_engineering import MouseFeatureEngineer


class MABeAdvancedDataset(Dataset):
    """Advanced dataset with feature engineering"""

    def __init__(self, data_dir, annotation_file=None, sequence_length=100,
                 frame_gap=1, train=True, transform=None,
                 use_feature_engineering=True, include_pca=False,
                 include_temporal=True):
        """
        Args:
            data_dir: Directory containing the keypoint data
            annotation_file: Path to annotation file (if available)
            sequence_length: Length of input sequences
            frame_gap: Gap between frames for temporal sampling
            train: Whether this is training data
            transform: Optional transform to be applied
            use_feature_engineering: Whether to apply feature engineering
            include_pca: Whether to include PCA features
            include_temporal: Whether to include temporal statistics
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.frame_gap = frame_gap
        self.train = train
        self.transform = transform
        self.use_feature_engineering = use_feature_engineering
        self.include_pca = include_pca
        self.include_temporal = include_temporal

        # Feature engineer
        self.feature_engineer = MouseFeatureEngineer(num_mice=2, num_keypoints=7)

        # Load data files
        self.data_files = sorted(list(self.data_dir.glob("*.npy")))

        # Load annotations if provided
        self.annotations = None
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)

        self.samples = self._prepare_samples()

        # Fit PCA if needed
        if self.use_feature_engineering and self.include_pca:
            self._fit_pca()

    def _prepare_samples(self):
        """Prepare list of (file_idx, start_frame) tuples"""
        samples = []

        for file_idx, file_path in enumerate(self.data_files):
            # Load keypoint data to get number of frames
            data = np.load(file_path)
            num_frames = len(data)

            # Create overlapping windows
            max_start = num_frames - (self.sequence_length * self.frame_gap)
            if max_start > 0:
                # During training, use sliding window with overlap
                if self.train:
                    stride = self.sequence_length // 4  # 75% overlap
                else:
                    stride = self.sequence_length

                for start_frame in range(0, max_start, stride):
                    samples.append((file_idx, start_frame))

        return samples

    def _fit_pca(self):
        """Fit PCA on all training data"""
        if not self.train:
            return

        print("Fitting PCA on training data...")

        # Collect all data
        all_data = []
        for file_path in self.data_files[:5]:  # Use first 5 files for PCA
            data = np.load(file_path)
            all_data.append(data)

        all_data = np.vstack(all_data)

        # Fit PCA
        self.feature_engineer.fit_pca(all_data, n_components=16)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start_frame = self.samples[idx]

        # Load keypoint data
        file_path = self.data_files[file_idx]
        data = np.load(file_path)

        # Extract sequence
        end_frame = start_frame + (self.sequence_length * self.frame_gap)
        sequence_raw = data[start_frame:end_frame:self.frame_gap]

        # Apply feature engineering if requested
        if self.use_feature_engineering:
            sequence = self.feature_engineer.extract_all_features(
                sequence_raw,
                include_pca=self.include_pca,
                include_temporal=self.include_temporal
            )
        else:
            sequence = sequence_raw

        # Convert to tensor
        sequence = torch.FloatTensor(sequence)

        # Get labels if available
        label = None
        if self.annotations:
            file_name = file_path.stem
            if file_name in self.annotations:
                # Extract labels for this sequence
                label = self._get_sequence_labels(
                    self.annotations[file_name],
                    start_frame,
                    end_frame,
                    self.frame_gap
                )
                label = torch.LongTensor(label)

        # Apply transforms (data augmentation)
        if self.transform:
            sequence = self.transform(sequence)

        if label is not None:
            return sequence, label
        else:
            return sequence

    def _get_sequence_labels(self, annotations, start_frame, end_frame, frame_gap):
        """Extract labels for a specific sequence"""
        labels = []
        for frame_idx in range(start_frame, end_frame, frame_gap):
            if frame_idx < len(annotations):
                labels.append(annotations[frame_idx])
            else:
                labels.append(0)  # Default label
        return labels


class DataAugmentation:
    """Data augmentation for mouse behavior sequences"""

    def __init__(self, noise_std=0.01, temporal_jitter=2):
        """
        Args:
            noise_std: Standard deviation of Gaussian noise
            temporal_jitter: Maximum frames to shift
        """
        self.noise_std = noise_std
        self.temporal_jitter = temporal_jitter

    def __call__(self, sequence):
        """
        Apply augmentation to sequence

        Args:
            sequence: [seq_len, features]

        Returns:
            augmented: [seq_len, features]
        """
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(sequence) * self.noise_std
            sequence = sequence + noise

        # Temporal jitter (random shift)
        if self.temporal_jitter > 0 and len(sequence) > self.temporal_jitter * 2:
            shift = np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1)
            if shift > 0:
                sequence = torch.cat([sequence[shift:], sequence[-shift:]], dim=0)
            elif shift < 0:
                sequence = torch.cat([sequence[:shift], sequence[:-shift]], dim=0)

        return sequence


def get_advanced_dataloaders(config):
    """Create train and validation dataloaders with advanced features"""

    # Create augmentation for training
    train_transform = None
    if config.get('use_augmentation', False):
        train_transform = DataAugmentation(
            noise_std=config.get('noise_std', 0.01),
            temporal_jitter=config.get('temporal_jitter', 2)
        )

    train_dataset = MABeAdvancedDataset(
        data_dir=config['train_data_dir'],
        annotation_file=config.get('train_annotation_file'),
        sequence_length=config['sequence_length'],
        frame_gap=config.get('frame_gap', 1),
        train=True,
        transform=train_transform,
        use_feature_engineering=config.get('use_feature_engineering', True),
        include_pca=config.get('include_pca', False),
        include_temporal=config.get('include_temporal_stats', True)
    )

    val_dataset = MABeAdvancedDataset(
        data_dir=config['val_data_dir'],
        annotation_file=config.get('val_annotation_file'),
        sequence_length=config['sequence_length'],
        frame_gap=config.get('frame_gap', 1),
        train=False,
        transform=None,  # No augmentation for validation
        use_feature_engineering=config.get('use_feature_engineering', True),
        include_pca=config.get('include_pca', False),
        include_temporal=config.get('include_temporal_stats', True)
    )

    # Copy PCA model from train to val if needed
    if config.get('use_feature_engineering') and config.get('include_pca'):
        val_dataset.feature_engineer.pca = train_dataset.feature_engineer.pca

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True  # For mixup
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader