#!/usr/bin/env python3
"""
Test the dataloader to verify data loading works correctly
"""

import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.dataset import MABeMouseDataset, get_dataloaders


def test_dataset():
    """Test dataset loading"""
    print("="*60)
    print("Testing MABe Dataset Loading")
    print("="*60)

    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nConfig:")
    print(f"  Train data dir: {config['train_data_dir']}")
    print(f"  Val data dir: {config['val_data_dir']}")
    print(f"  Sequence length: {config['sequence_length']}")
    print(f"  Batch size: {config['batch_size']}")

    # Create dataset
    print("\n" + "-"*60)
    print("Creating training dataset...")
    train_dataset = MABeMouseDataset(
        data_dir=config['train_data_dir'],
        annotation_file=config.get('train_annotation_file'),
        sequence_length=config['sequence_length'],
        frame_gap=config.get('frame_gap', 1),
        train=True
    )

    print(f"  Number of samples: {len(train_dataset)}")
    print(f"  Number of data files: {len(train_dataset.data_files)}")

    # Test getting a sample
    print("\nTesting sample retrieval...")
    sample = train_dataset[0]

    if isinstance(sample, tuple):
        sequence, label = sample
        print(f"  Sequence shape: {sequence.shape}")
        print(f"  Label shape: {label.shape}")
        print(f"  Sequence dtype: {sequence.dtype}")
        print(f"  Label dtype: {label.dtype}")
        print(f"  Sample labels (first 10): {label[:10].tolist()}")
    else:
        print(f"  Sequence shape: {sample.shape}")
        print(f"  No labels available")

    # Create validation dataset
    print("\n" + "-"*60)
    print("Creating validation dataset...")
    val_dataset = MABeMouseDataset(
        data_dir=config['val_data_dir'],
        annotation_file=config.get('val_annotation_file'),
        sequence_length=config['sequence_length'],
        frame_gap=config.get('frame_gap', 1),
        train=False
    )

    print(f"  Number of samples: {len(val_dataset)}")
    print(f"  Number of data files: {len(val_dataset.data_files)}")

    # Test dataloaders
    print("\n" + "-"*60)
    print("Testing dataloaders...")
    train_loader, val_loader = get_dataloaders(config)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Test loading a batch
    print("\nLoading first batch...")
    batch = next(iter(train_loader))

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        inputs, targets = batch
        print(f"  Input batch shape: {inputs.shape}")
        print(f"  Target batch shape: {targets.shape}")
        print(f"  Input dtype: {inputs.dtype}")
        print(f"  Target dtype: {targets.dtype}")
    else:
        print(f"  Batch type: {type(batch)}")
        if hasattr(batch, 'shape'):
            print(f"  Batch shape: {batch.shape}")

    print("\n" + "="*60)
    print("Dataloader test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    test_dataset()