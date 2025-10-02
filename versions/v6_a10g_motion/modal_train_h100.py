"""
Train on real Kaggle MABe data using Modal H100
Ultra-fast training with motion features (speed + acceleration)
Expected time: ~1.4 hours for 100 epochs
"""

import modal

app = modal.App("mabe-h100-training")
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
    gpu="H100",  # 80GB VRAM, 267 TFLOPS
    timeout=3600 * 4,  # 4 hours (should finish in ~1.4h)
    volumes={"/vol": volume},
    memory=65536,  # 64GB RAM for large batch
)
def train_h100_model(
    config_name: str = "config_h100.yaml",
    max_sequences: int = None,  # None = use all data
):
    """Train model on H100 GPU"""
    import torch
    import yaml
    import sys
    from pathlib import Path

    print("="*60)
    print("MABe Kaggle Data Training on Modal H100")
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
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")

    print(f"Model: {config['model_type']}")
    print(f"Batch Size: {config.get('batch_size', 384)} (H100 Optimized)")
    print(f"Motion Features: Enabled (speed + acceleration)")
    print(f"Using Kaggle real data: True")
    print()

    # Import modules
    from src.data.kaggle_dataset import create_kaggle_dataloaders
    from src.models.advanced_models import Conv1DBiLSTM, TemporalConvNet, HybridModel
    from src.utils.advanced_trainer import AdvancedTrainer

    # Create dataloaders
    print("Loading Kaggle data...")
    train_loader, val_loader = create_kaggle_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config.get('batch_size', 384),
        sequence_length=config.get('sequence_length', 100),
        num_workers=2,  # Modal limitation
        use_feature_engineering=False,
        max_sequences=max_sequences,  # None = all data
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

    # Update checkpoint directory to Modal volume
    config['checkpoint_dir'] = '/vol/checkpoints/h100'
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
    print("Starting training on H100...")
    print("="*60)
    print(f"Expected time: ~1.4 hours (100 epochs)")
    print(f"Batch size: {config.get('batch_size', 384)}")
    print(f"Estimated speed: ~8x faster than A10G")
    print("="*60)
    print()

    import time
    start_time = time.time()

    trainer.train()

    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600

    # Final commit to save all checkpoints
    print(f"\n⏳ Final volume commit...")
    volume.commit()

    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)
    print(f"Training time: {hours:.2f} hours")
    print(f"Speed vs A10G: {12/hours:.1f}x faster")
    print(f"Checkpoints saved to Modal volume: /vol/checkpoints/h100/")
    print("="*60)

    return {
        'best_val_loss': trainer.best_val_loss,
        'best_val_f1': trainer.best_val_f1,
        'total_epochs': trainer.current_epoch + 1,
        'training_time_hours': hours,
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def download_checkpoint(checkpoint_name: str = "best_model.pth"):
    """Download checkpoint from Modal volume"""
    from pathlib import Path

    checkpoint_path = Path("/vol/checkpoints/h100") / checkpoint_name

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None

    with open(checkpoint_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    config: str = "config_h100.yaml",
    max_sequences: int = None,
):
    """
    Train model on H100 GPU

    Args:
        config: Config file name
        max_sequences: Maximum sequences to load (None = all data, default)
    """
    print("\n" + "="*60)
    print("MABe Kaggle Training - H100 GPU (Ultra-Fast)")
    print("="*60)
    print(f"Config: {config}")
    if max_sequences:
        print(f"⚠️  Limited mode: Max sequences = {max_sequences}")
    else:
        print(f"✓ Full dataset mode: Using ALL available videos (~863)")
    print(f"✓ Motion features: Enabled (speed + acceleration)")
    print(f"✓ Expected time: ~1.4 hours")
    print("="*60 + "\n")

    # Train model
    result = train_h100_model.remote(
        config_name=config,
        max_sequences=max_sequences,
    )

    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)
    print(f"Best Val Loss: {result['best_val_loss']:.4f}")
    print(f"Best Val F1: {result['best_val_f1']:.4f}")
    print(f"Total Epochs: {result['total_epochs']}")
    print(f"Training Time: {result['training_time_hours']:.2f} hours")
    print(f"Speed: {12/result['training_time_hours']:.1f}x faster than A10G")
    print("="*60)


if __name__ == "__main__":
    main()
