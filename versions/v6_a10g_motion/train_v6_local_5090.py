"""
V6 Local Training for RTX 5090
Train with motion features (speed + acceleration) on local GPU
"""
import torch
import yaml
import sys
from pathlib import Path
import argparse

def train_v6_local(
    config_path: str = "configs/config_5090.yaml",
    data_dir: str = "data/kaggle",
    checkpoint_dir: str = "checkpoints/v6_5090",
    resume_from: str = None,
):
    """
    Train V6 model on local RTX 5090

    Args:
        config_path: Path to config file
        data_dir: Path to Kaggle data directory
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    print("="*60)
    print("MABe V6 Training on RTX 5090")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Data Dir: {data_dir}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print()

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update paths
    config['data_dir'] = data_dir
    config['checkpoint_dir'] = checkpoint_dir
    config['use_kaggle_data'] = True

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Import modules
    from src.data.kaggle_dataset import create_kaggle_dataloaders
    from src.models.advanced_models import Conv1DBiLSTM
    from src.utils.advanced_trainer import AdvancedTrainer

    # Create dataloaders
    print("Loading Kaggle data with motion features...")
    print(f"  Batch size: {config.get('batch_size', 64)}")
    print(f"  Sequence length: {config.get('sequence_length', 100)}")
    print(f"  Num workers: {config.get('num_workers', 8)}")
    print()

    train_loader, val_loader = create_kaggle_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config.get('batch_size', 64),
        sequence_length=config.get('sequence_length', 100),
        num_workers=config.get('num_workers', 8),
        use_feature_engineering=False,  # Motion features added automatically
        max_sequences=None,  # Use all data
    )

    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")

    # Get input dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"✓ Input dimension: {input_dim}")
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
    print(f"✓ Model: Conv1DBiLSTM")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Conv channels: {config.get('conv_channels', [64, 128, 256])}")
    print(f"✓ LSTM hidden: {config.get('lstm_hidden', 256)}")
    print(f"✓ LSTM layers: {config.get('lstm_layers', 2)}")
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

    # Resume from checkpoint if specified
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"  Resumed from epoch {checkpoint['epoch']}")
        print(f"  Best Val F1: {trainer.best_val_f1:.4f}")
        print()

    # Train
    print("="*60)
    print("Starting training...")
    print("="*60)
    print()

    trainer.train()

    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)
    print(f"Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"Best Val F1: {trainer.best_val_f1:.4f}")
    print(f"Total Epochs: {trainer.current_epoch}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*60)

    return {
        'best_val_loss': trainer.best_val_loss,
        'best_val_f1': trainer.best_val_f1,
        'total_epochs': trainer.current_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description='Train V6 model on RTX 5090')
    parser.add_argument('--config', type=str, default='configs/config_5090.yaml',
                       help='Path to config file')
    parser.add_argument('--data-dir', type=str, default='data/kaggle',
                       help='Path to Kaggle data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/v6_5090',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    result = train_v6_local(
        config_path=args.config,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
    )

    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    print(f"Best Validation F1: {result['best_val_f1']:.4f}")
    print(f"Best Validation Loss: {result['best_val_loss']:.4f}")
    print(f"Total Epochs Trained: {result['total_epochs']}")
    print("="*60)


if __name__ == "__main__":
    main()
