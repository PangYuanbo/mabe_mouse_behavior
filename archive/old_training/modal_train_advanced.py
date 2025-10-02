"""
Modal deployment script for MABe with advanced features
"""

import modal

# Create Modal app
app = modal.App("mabe-advanced")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.7.0",
        "kaggle>=1.7.0",
    )
)

# Create a volume to persist data and checkpoints
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Upgraded GPU
    timeout=3600 * 6,  # 6 hours timeout
    volumes={"/vol": volume},
)
def train_advanced_model(config_name: str = "config_advanced.yaml"):
    """
    Train MABe mouse behavior detection model with advanced features

    Args:
        config_name: Name of config file in /vol/code/configs/
    """
    import os
    import sys
    import yaml
    import torch
    from pathlib import Path

    # Add code directory to path
    sys.path.insert(0, '/vol/code')

    # Import training modules
    from src.data.advanced_dataset import get_advanced_dataloaders
    from src.models.advanced_models import build_advanced_model
    from src.utils.advanced_trainer import AdvancedTrainer

    # Load configuration from volume
    config_file = Path(f"/vol/code/configs/{config_name}")
    print(f"Loading config from {config_file}...")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths to use volume storage
    config['train_data_dir'] = '/vol/train'
    config['val_data_dir'] = '/vol/val'
    config['train_annotation_file'] = '/vol/train_annotations.json'
    config['val_annotation_file'] = '/vol/val_annotations.json'
    config['checkpoint_dir'] = '/vol/checkpoints_advanced'
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {config['device']}")
    print(f"Model: {config['model_type']}")
    print(f"Feature Engineering: {config.get('use_feature_engineering', True)}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = get_advanced_dataloaders(config)
    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")

    # Get actual input dim
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_input = sample_batch[0]
    else:
        sample_input = sample_batch

    actual_input_dim = sample_input.shape[-1]
    config['input_dim'] = actual_input_dim
    print(f"✓ Input dimension: {actual_input_dim}")

    # Build model
    print("\nBuilding model...")
    model = build_advanced_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {config['model_type']}")
    print(f"✓ Total parameters: {num_params:,}")

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device']
    )

    # Train
    print("\n" + "="*60)
    print("Starting training on Modal...")
    print("="*60)
    trainer.train()

    # Commit volume to persist checkpoints
    volume.commit()

    print("\n" + "="*60)
    print("Training complete!")
    print(f"✓ Best model saved to: /vol/checkpoints_advanced/best_model.pth")
    print(f"✓ Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"✓ Best validation loss: {trainer.best_val_loss:.4f}")
    print("="*60)

    return {
        "best_val_loss": trainer.best_val_loss,
        "best_val_f1": trainer.best_val_f1,
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
        "val_metrics": trainer.val_metrics,
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def download_checkpoint(checkpoint_name: str = "best_model.pth", checkpoint_dir: str = "checkpoints_advanced"):
    """
    Download a checkpoint from Modal volume

    Args:
        checkpoint_name: Name of checkpoint file to download
        checkpoint_dir: Directory containing checkpoint
    """
    from pathlib import Path

    checkpoint_path = Path(f"/vol/{checkpoint_dir}/{checkpoint_name}")

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint {checkpoint_name} not found in {checkpoint_dir}")
        return None

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = f.read()

    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    return checkpoint_data


@app.local_entrypoint()
def main(config: str = "config_advanced.yaml"):
    """
    Main entry point for Modal training

    Usage:
        modal run modal_train_advanced.py
        modal run modal_train_advanced.py --config config_custom.yaml
    """
    print("="*60)
    print("MABe Advanced Training on Modal")
    print("="*60)
    print(f"Config: {config}")

    # Train model
    result = train_advanced_model.remote(config_name=config)

    print("\n" + "="*60)
    print("Training Results:")
    print("="*60)
    print(f"Best validation F1: {result['best_val_f1']:.4f}")
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    print(f"Final train loss: {result['train_losses'][-1]:.4f}")
    print(f"Final val loss: {result['val_losses'][-1]:.4f}")

    if result['val_metrics']:
        print("\nFinal validation metrics:")
        for metric_name, metric_value in result['val_metrics'][-1].items():
            print(f"  {metric_name}: {metric_value:.4f}")

    print("\n✓ Checkpoints saved to Modal volume 'mabe-data'")
    print("To download: modal run modal_train_advanced.py::download_checkpoint")


if __name__ == "__main__":
    main()