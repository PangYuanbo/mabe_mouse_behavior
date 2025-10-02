"""
Train V6 on Modal A10G (24GB VRAM)
More cost-effective than H100 for this model size
"""
import modal

app = modal.App("mabe-v6-a10g-training")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "numpy==2.3.3",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
        "scikit-learn==1.7.2",
        "scipy==1.16.2",
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
    )
)


@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM - perfect for this model
    timeout=3600 * 12,  # 12 hours
    volumes={"/vol": volume},
    memory=32768,  # 32GB RAM
)
def train_v6_a10g(
    config_name: str = "config_v6_a10g.yaml",
    max_sequences: int = None,  # None = use all data
):
    """Train V6 model on A10G GPU"""
    import torch
    import yaml
    import sys
    from pathlib import Path

    print("="*60)
    print("MABe V6 Training on Modal A10G")
    print("="*60)
    print(f"Config: {config_name}")

    # Add code directory to path
    sys.path.insert(0, "/vol/code")

    # Load config
    config_path = Path("/vol/code/configs") / config_name
    print(f"Loading config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update config for Kaggle data
    config['data_dir'] = '/vol/data/kaggle'
    config['use_kaggle_data'] = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    print(f"Model: {config['model_type']}")
    print(f"Batch Size: {config.get('batch_size', 384)}")
    print(f"Motion Features: Enabled (speed + acceleration)")
    print(f"Using Kaggle real data: True")
    print()

    # Import modules
    from src.data.kaggle_dataset import create_kaggle_dataloaders
    from src.models.advanced_models import Conv1DBiLSTM
    from src.utils.advanced_trainer import AdvancedTrainer

    # Create dataloaders
    print("Loading Kaggle data with motion features...")
    train_loader, val_loader = create_kaggle_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config.get('batch_size', 384),
        sequence_length=config.get('sequence_length', 100),
        num_workers=4,  # Increased from 2
        use_feature_engineering=False,
        max_sequences=max_sequences,
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
    print("Building model...")
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
    print()

    # Update checkpoint directory
    config['checkpoint_dir'] = '/vol/checkpoints/v6_a10g'
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    # Define epoch callback for periodic volume commits
    def epoch_callback(epoch):
        """Commit volume every 5 epochs to persist checkpoints"""
        if epoch % 5 == 0:
            print(f"\n⏳ Committing volume at epoch {epoch}...")
            volume.commit()
            print(f"✓ Volume committed successfully")

    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        epoch_callback=epoch_callback,
    )

    # Train
    print("="*60)
    print("Starting training on A10G...")
    print("="*60)

    trainer.train()

    # Final commit to save all checkpoints
    print("\n⏳ Final volume commit...")
    volume.commit()

    print("\n✓ Training complete! Checkpoints saved to Modal volume.")

    return {
        'best_val_loss': trainer.best_val_loss,
        'best_val_f1': trainer.best_val_f1,
        'total_epochs': trainer.current_epoch + 1,
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def download_checkpoint(checkpoint_name: str = "best_model.pth"):
    """Download checkpoint from Modal volume"""
    from pathlib import Path

    checkpoint_paths = [
        Path("/vol/checkpoints/v6_a10g") / checkpoint_name,
        Path("/vol/checkpoints/h100") / checkpoint_name,
        Path("/vol/checkpoints") / checkpoint_name,
    ]

    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            print(f"Found checkpoint: {checkpoint_path}")
            with open(checkpoint_path, "rb") as f:
                return f.read()

    print(f"✗ Checkpoint not found: {checkpoint_name}")
    return None


@app.local_entrypoint()
def main(
    config: str = "config_v6_a10g.yaml",
    max_sequences: int = None,
):
    """
    Train V6 model on A10G

    Args:
        config: Config file name
        max_sequences: Maximum sequences to load (None = all data)
    """
    print("\n" + "="*60)
    print("MABe V6 Training - A10G GPU")
    print("="*60)
    print(f"Config: {config}")
    if max_sequences:
        print(f"⚠️  Limited mode: Max sequences = {max_sequences}")
    else:
        print(f"✓ Full dataset mode: Using ALL available videos (~863)")
    print(f"✓ Motion features: Enabled (288-dim input)")
    print("="*60 + "\n")

    # Train model
    result = train_v6_a10g.remote(
        config_name=config,
        max_sequences=max_sequences,
    )

    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Best Val F1: {result['best_val_f1']:.4f}")
    print(f"Total Epochs: {result['total_epochs']}")
    print("="*60)


if __name__ == "__main__":
    main()
