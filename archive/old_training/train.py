#!/usr/bin/env python3
"""
Main training script for MABe Mouse Behavior Detection
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

from data.dataset import get_dataloaders
from models.transformer_model import build_model
from utils.trainer import Trainer


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
    parser = argparse.ArgumentParser(description='Train MABe Mouse Behavior Detection Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
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
        print("CUDA not available, using CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    # Create dataloaders
    print("\nLoading data...")
    try:
        train_loader, val_loader = get_dataloaders(config)
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nMake sure to set correct paths in the config file:")
        print(f"  - train_data_dir: {config['train_data_dir']}")
        print(f"  - val_data_dir: {config['val_data_dir']}")
        return

    # Build model
    print("\nBuilding model...")
    model = build_model(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {config['model_type']}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    trainer.train()

    print("\nTraining complete!")
    print(f"Best model saved to: {trainer.checkpoint_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()