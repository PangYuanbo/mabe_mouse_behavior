#!/usr/bin/env python3
"""
Advanced training script for MABe Mouse Behavior Detection
Uses feature engineering and improved training strategies
"""

import argparse
import yaml
import torch
import random
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.advanced_dataset import get_advanced_dataloaders
from models.advanced_models import build_advanced_model
from utils.advanced_trainer import AdvancedTrainer


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train MABe Mouse Behavior Detection Model (Advanced)')
    parser.add_argument('--config', type=str, default='configs/config_advanced.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override device if specified
    if args.device is not None:
        config['device'] = args.device

    # Set device
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'

    print("="*60)
    print("MABe Mouse Behavior Detection - Advanced Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model: {config.get('model_type', 'conv_bilstm')}")
    print(f"Feature Engineering: {config.get('use_feature_engineering', True)}")
    print(f"Data Augmentation: {config.get('use_augmentation', False)}")

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Create dataloaders
    print("\nLoading data...")
    try:
        train_loader, val_loader = get_advanced_dataloaders(config)
        print(f"✓ Training samples: {len(train_loader.dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")

        # Get actual input dim from dataset
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0]
        else:
            sample_input = sample_batch

        actual_input_dim = sample_input.shape[-1]
        print(f"✓ Input dimension: {actual_input_dim}")

        # Update config with actual input dim
        config['input_dim'] = actual_input_dim

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nMake sure to set correct paths in the config file:")
        print(f"  - train_data_dir: {config['train_data_dir']}")
        print(f"  - val_data_dir: {config['val_data_dir']}")
        import traceback
        traceback.print_exc()
        return

    # Build model
    print("\nBuilding model...")
    model = build_advanced_model(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model: {config['model_type']}")
    print(f"✓ Total parameters: {num_params:,}")
    print(f"✓ Trainable parameters: {num_trainable:,}")

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\n✓ Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training configuration summary
    print("\n" + "="*60)
    print("Training Configuration:")
    print("="*60)
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch size: {config.get('batch_size', 32)}")
    print(f"Learning rate: {config.get('learning_rate', 0.0005)}")
    print(f"Optimizer: {config.get('optimizer', 'adamw')}")
    print(f"Scheduler: {config.get('scheduler', 'plateau')}")
    print(f"Loss: {config.get('loss', 'cross_entropy')}")

    if config.get('class_weights'):
        print(f"Class weights: {config.get('class_weights')}")
    if config.get('label_smoothing', 0) > 0:
        print(f"Label smoothing: {config.get('label_smoothing')}")
    if config.get('mixup_alpha', 0) > 0:
        print(f"Mixup alpha: {config.get('mixup_alpha')}")

    print(f"Gradient clipping: {config.get('grad_clip', 1.0)}")
    print(f"Early stopping patience: {config.get('early_stopping_patience', 15)}")

    # Start training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(is_best=False)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"✓ Best model saved to: {trainer.checkpoint_dir / 'best_model.pth'}")
    print(f"✓ Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"✓ Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == '__main__':
    main()