"""
Train on real Kaggle MABe data using local RTX 5090
With motion features (speed + acceleration)
"""

import torch
import yaml
import argparse
from pathlib import Path
import os

def train_local(
    config_path: str = "configs/config_5090.yaml",
    data_dir: str = "data/kaggle",
    max_sequences: int = None,
):
    """Train model on local RTX 5090"""

    print("="*60)
    print("MABe Kaggle Data Training - Local RTX 5090")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Data dir: {data_dir}")
    if max_sequences:
        print(f"Max sequences: {max_sequences}")
    print("="*60)
    print()

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update config
    config['data_dir'] = data_dir
    config['use_kaggle_data'] = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"Model: {config['model_type']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Motion features: Enabled (speed + acceleration)")
    print()

    # Import modules
    from src.data.kaggle_dataset import create_kaggle_dataloaders
    from src.models.advanced_models import Conv1DBiLSTM, TemporalConvNet, HybridModel
    from src.utils.advanced_trainer import AdvancedTrainer

    # Create dataloaders
    print("Loading Kaggle data...")
    train_loader, val_loader = create_kaggle_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config.get('batch_size', 96),
        sequence_length=config.get('sequence_length', 100),
        num_workers=config.get('num_workers', 4),
        use_feature_engineering=False,
        max_sequences=max_sequences,
    )

    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")

    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"✓ Input dimension: {input_dim}")
    print(f"  - Original coordinates: 144")
    print(f"  - Speed features: 72")
    print(f"  - Acceleration features: 72")
    print()

    # Update config with actual input dimension
    config['input_dim'] = input_dim

    # Build model
    print("Building model...")
    model_type = config.get('model_type', 'conv_bilstm')

    if model_type == 'conv_bilstm':
        model = Conv1DBiLSTM(
            input_dim=input_dim,
            num_classes=config['num_classes'],
            conv_channels=config.get('conv_channels', [64, 128, 256]),
            lstm_hidden=config.get('lstm_hidden', 256),
            lstm_layers=config.get('lstm_layers', 2),
            dropout=config.get('dropout', 0.3),
        )
    elif model_type == 'tcn':
        model = TemporalConvNet(
            input_dim=input_dim,
            num_classes=config['num_classes'],
            num_channels=config.get('tcn_channels', [64, 128, 256, 256]),
            kernel_size=config.get('tcn_kernel_size', 3),
            dropout=config.get('dropout', 0.3),
        )
    elif model_type == 'hybrid':
        model = HybridModel(
            input_dim=input_dim,
            num_classes=config['num_classes'],
            num_keypoints=config.get('num_keypoints', 18),
            pointnet_dim=config.get('pointnet_dim', 128),
            temporal_model=config.get('temporal_model', 'lstm'),
            lstm_hidden=config.get('lstm_hidden', 256),
            lstm_layers=config.get('lstm_layers', 2),
            dropout=config.get('dropout', 0.3),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {model_type}")
    print(f"✓ Total parameters: {total_params:,}")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints/5090'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config['checkpoint_dir'] = str(checkpoint_dir)

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    # Train
    print("="*60)
    print("Starting training on RTX 5090...")
    print("="*60)
    print(f"Epochs: {config['epochs']}")
    print(f"Early stopping patience: {config.get('early_stopping_patience', 15)}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print("="*60)
    print()

    trainer.train()

    print("\n✓ Training complete!")
    print(f"Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"Best Val F1: {trainer.best_val_f1:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MABe model on local RTX 5090")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_5090.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kaggle",
        help="Path to Kaggle data directory"
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to load (for testing)"
    )

    args = parser.parse_args()

    train_local(
        config_path=args.config,
        data_dir=args.data_dir,
        max_sequences=args.max_sequences,
    )
