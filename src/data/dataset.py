import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path


class MABeMouseDataset(Dataset):
    """Dataset for MABe mouse behavior detection"""

    def __init__(self, data_dir, annotation_file=None, sequence_length=64,
                 frame_gap=1, train=True, transform=None):
        """
        Args:
            data_dir: Directory containing the keypoint data
            annotation_file: Path to annotation file (if available)
            sequence_length: Length of input sequences
            frame_gap: Gap between frames for temporal sampling
            train: Whether this is training data
            transform: Optional transform to be applied
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.frame_gap = frame_gap
        self.train = train
        self.transform = transform

        # Load data files
        self.data_files = sorted(list(self.data_dir.glob("*.npy")))

        # Load annotations if provided
        self.annotations = None
        if annotation_file and os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)

        self.samples = self._prepare_samples()

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start_frame = self.samples[idx]

        # Load keypoint data
        file_path = self.data_files[file_idx]
        data = np.load(file_path)

        # Extract sequence
        end_frame = start_frame + (self.sequence_length * self.frame_gap)
        sequence = data[start_frame:end_frame:self.frame_gap]

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

        # Apply transforms
        if self.transform:
            sequence = self.transform(sequence)

        if label is not None:
            return sequence, label
        else:
            return sequence

    def _get_sequence_labels(self, annotations, start_frame, end_frame, frame_gap):
        """Extract labels for a specific sequence"""
        # This should be adapted based on the actual annotation format
        labels = []
        for frame_idx in range(start_frame, end_frame, frame_gap):
            if frame_idx < len(annotations):
                labels.append(annotations[frame_idx])
            else:
                labels.append(0)  # Default label
        return labels


def get_dataloaders(config):
    """Create train and validation dataloaders"""

    train_dataset = MABeMouseDataset(
        data_dir=config['train_data_dir'],
        annotation_file=config.get('train_annotation_file'),
        sequence_length=config['sequence_length'],
        frame_gap=config.get('frame_gap', 1),
        train=True
    )

    val_dataset = MABeMouseDataset(
        data_dir=config['val_data_dir'],
        annotation_file=config.get('val_annotation_file'),
        sequence_length=config['sequence_length'],
        frame_gap=config.get('frame_gap', 1),
        train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    return train_loader, val_loader