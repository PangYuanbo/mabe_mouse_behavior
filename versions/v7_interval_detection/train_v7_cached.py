"""
V7 Training with Cached Windows - Ultra Fast Startup
Run from project root: python train_v7_cached.py

Prerequisites:
1. Run preprocess_windows.py first to generate cached windows
"""

import torch
import yaml
import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.cached_dataset import create_cached_dataloaders
from src.models.advanced_models import Conv1DBiLSTM
from src.utils.advanced_trainer import AdvancedTrainer


def main():
    parser = argparse.ArgumentParser(description='V7 Training - Cached Windows')
    parser.add_argument('--config', type=str, default='configs/config_v7_5090.yaml',
                       help='Path to config file')
    parser.add_argument('--train_cache', type=str,
                       default='data/kaggle/preprocessed_windows/train_windows_seq100_stride25.pkl',
                       help='Path to training windows cache')
    parser.add_argument('--val_cache', type=str,
                       default='data/kaggle/preprocessed_windows/val_windows_seq100_stride100.pkl',
                       help='Path to validation windows cache')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"V7 Training - Cached Windows (Fast Startup)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create dataloaders from cached windows
    print(f"\nLoading cached windows...")
    print(f"  Train cache: {args.train_cache}")
    print(f"  Val cache: {args.val_cache}")
    print()

    train_loader, val_loader = create_cached_dataloaders(
        train_cache_file=args.train_cache,
        val_cache_file=args.val_cache,
        batch_size=config.get('batch_size', 384),
        num_workers=config.get('num_workers', 8),
        pin_memory=config.get('pin_memory', True),
        prefetch_factor=config.get('prefetch_factor', 2),
    )

    print(f"[OK] Training batches: {len(train_loader)}")
    print(f"[OK] Validation batches: {len(val_loader)}")

    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"[OK] Input dimension: {input_dim}")
    print(f"  - Original coordinates: 144 (4 mice × 18 keypoints × 2)")
    print(f"  - Speed features: 72 (4 mice × 18 keypoints)")
    print(f"  - Acceleration features: 72 (4 mice × 18 keypoints)")
    print()

    # Update config with actual input dimension
    config['input_dim'] = input_dim

    # Build model
    print("Building Conv1DBiLSTM model...")
    model = Conv1DBiLSTM(
        input_dim=input_dim,
        num_classes=config['num_classes'],
        conv_channels=config.get('conv_channels', [64, 128, 256]),
        lstm_hidden=config.get('lstm_hidden', 256),
        lstm_layers=config.get('lstm_layers', 2),
        dropout=config.get('dropout', 0.3),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model: Conv1DBiLSTM")
    print(f"[OK] Total parameters: {total_params:,}")
    print(f"[OK] Conv channels: {config.get('conv_channels', [64, 128, 256])}")
    print(f"[OK] LSTM hidden: {config.get('lstm_hidden', 256)}")
    print(f"[OK] LSTM layers: {config.get('lstm_layers', 2)}")
    print()

    # Create trainer
    print("="*60)
    print("Initializing trainer...")
    print("="*60)

    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Train
    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Checkpoint dir: {config['checkpoint_dir']}")
    print()

    trainer.train(epochs=config['epochs'])

    print(f"\n{'='*60}")
    print(f"[OK] Training complete!")
    print(f"  Best F1: {trainer.best_val_f1:.4f}")
    print(f"  Checkpoint: {trainer.checkpoint_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
