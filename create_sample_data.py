#!/usr/bin/env python3
"""
Create sample data for testing the MABe training framework
"""

import numpy as np
import json
from pathlib import Path


def create_sample_data():
    """Create sample keypoint data and annotations"""

    # Create directories
    data_dir = Path('data')
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Parameters based on MABe competition
    # MABe uses pose estimation with multiple keypoints per mouse
    num_mice = 2  # Dyadic interactions
    num_keypoints_per_mouse = 7  # Head, ears, spine points, tail base
    num_coords = 2  # x, y coordinates

    # Total features per frame: num_mice * num_keypoints * num_coords
    features_per_frame = num_mice * num_keypoints_per_mouse * num_coords  # 28

    # Behavior classes for MABe
    behavior_classes = {
        0: 'other',
        1: 'close_investigation',
        2: 'mount',
        3: 'attack'
    }
    num_classes = len(behavior_classes)

    print("Creating sample MABe mouse behavior data...")
    print(f"Features per frame: {features_per_frame}")
    print(f"Number of behavior classes: {num_classes}")
    print(f"Behavior classes: {behavior_classes}")

    # Create training data
    print("\nGenerating training data...")
    train_annotations = {}

    for i in range(5):  # 5 training sequences
        # Random number of frames (between 500-2000 frames ~30-60 seconds at 30fps)
        num_frames = np.random.randint(500, 2000)

        # Generate keypoint data
        # Shape: (num_frames, num_mice, num_keypoints, 2)
        # Normalized coordinates between 0 and 1
        data = np.random.rand(num_frames, num_mice, num_keypoints_per_mouse, num_coords)

        # Add some temporal coherence (smooth motion)
        for t in range(1, num_frames):
            data[t] = 0.7 * data[t] + 0.3 * data[t-1]

        # Reshape to (num_frames, features)
        data_flat = data.reshape(num_frames, -1)

        # Save as npy file
        filename = f'train_video_{i:03d}'
        np.save(train_dir / f'{filename}.npy', data_flat.astype(np.float32))

        # Generate frame-level annotations
        # Random behavior labels for each frame
        labels = np.random.choice(num_classes, size=num_frames, p=[0.7, 0.15, 0.1, 0.05])
        train_annotations[filename] = labels.tolist()

        print(f"  Created {filename}.npy: {num_frames} frames")

    # Save training annotations
    with open(data_dir / 'train_annotations.json', 'w') as f:
        json.dump(train_annotations, f, indent=2)

    print(f"\nSaved training annotations to {data_dir / 'train_annotations.json'}")

    # Create validation data
    print("\nGenerating validation data...")
    val_annotations = {}

    for i in range(2):  # 2 validation sequences
        num_frames = np.random.randint(500, 1000)

        data = np.random.rand(num_frames, num_mice, num_keypoints_per_mouse, num_coords)

        # Add temporal coherence
        for t in range(1, num_frames):
            data[t] = 0.7 * data[t] + 0.3 * data[t-1]

        data_flat = data.reshape(num_frames, -1)

        filename = f'val_video_{i:03d}'
        np.save(val_dir / f'{filename}.npy', data_flat.astype(np.float32))

        labels = np.random.choice(num_classes, size=num_frames, p=[0.7, 0.15, 0.1, 0.05])
        val_annotations[filename] = labels.tolist()

        print(f"  Created {filename}.npy: {num_frames} frames")

    # Save validation annotations
    with open(data_dir / 'val_annotations.json', 'w') as f:
        json.dump(val_annotations, f, indent=2)

    print(f"\nSaved validation annotations to {data_dir / 'val_annotations.json'}")

    # Create metadata file
    metadata = {
        'dataset': 'MABe Mouse Behavior (Sample)',
        'num_mice': num_mice,
        'num_keypoints_per_mouse': num_keypoints_per_mouse,
        'num_coords': num_coords,
        'features_per_frame': features_per_frame,
        'num_classes': num_classes,
        'behavior_classes': behavior_classes,
        'train_sequences': len(train_annotations),
        'val_sequences': len(val_annotations)
    }

    with open(data_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {data_dir / 'metadata.json'}")
    print("\n" + "="*60)
    print("Sample data creation complete!")
    print("="*60)
    print(f"\nDataset structure:")
    print(f"  data/train/: {len(train_annotations)} sequences")
    print(f"  data/val/: {len(val_annotations)} sequences")
    print(f"  Features per frame: {features_per_frame}")
    print(f"  Behavior classes: {num_classes}")
    print(f"\nUpdate configs/config.yaml with:")
    print(f"  input_dim: {features_per_frame}")
    print(f"  num_classes: {num_classes}")


if __name__ == '__main__':
    create_sample_data()