"""
Modal deployment script for MABe Mouse Behavior Detection training
"""

import modal

# Create Modal app
app = modal.App("mabe-mouse-behavior")

# Define the container image with all dependencies
# Note: Source code will be uploaded inline with the function
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "kaggle>=1.7.0",
    )
)

# Create a volume to persist data and checkpoints
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

# Kaggle credentials secret
# To set: modal secret create kaggle-credentials KAGGLE_USERNAME=<username> KAGGLE_KEY=<key>


@app.function(
    image=image,
    gpu="T4",  # Use T4 GPU
    timeout=3600 * 4,  # 4 hours timeout
    volumes={"/vol": volume},
)
def train_model(
    config_path: str = "configs/config.yaml",
    download_data: bool = False,
):
    """
    Train MABe mouse behavior detection model on Modal

    Args:
        config_path: Path to training configuration file
        download_data: Whether to download data from Kaggle
    """
    import os
    import sys
    import subprocess
    import yaml
    import torch
    from pathlib import Path

    # Set up Kaggle credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        kaggle_json = {
            "username": os.environ["KAGGLE_USERNAME"],
            "key": os.environ["KAGGLE_KEY"]
        }
        import json
        with open(kaggle_dir / "kaggle.json", "w") as f:
            json.dump(kaggle_json, f)
        os.chmod(kaggle_dir / "kaggle.json", 0o600)
        print("✓ Kaggle credentials configured")

    # Download data if requested
    if download_data:
        print("\nDownloading MABe competition data from Kaggle...")
        data_dir = Path("/data")
        data_dir.mkdir(exist_ok=True)

        try:
            subprocess.run([
                "kaggle", "competitions", "download",
                "-c", "mabe-mouse-behavior-detection",
                "-p", str(data_dir)
            ], check=True)

            # Unzip data
            import zipfile
            zip_path = data_dir / "mabe-mouse-behavior-detection.zip"
            if zip_path.exists():
                print("Extracting data...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print("✓ Data downloaded and extracted")
        except Exception as e:
            print(f"Warning: Could not download data: {e}")
            print("Using existing data in /data volume")

    # Add code directory to path
    sys.path.insert(0, '/vol/code')

    # Import training modules
    from src.data.dataset import get_dataloaders
    from src.models.transformer_model import build_model
    from src.utils.trainer import Trainer

    # Load configuration from volume
    config_file = Path("/vol/code/configs/config.yaml")
    print(f"\nLoading config from {config_file}...")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths to use volume storage
    config['train_data_dir'] = '/vol/train'
    config['val_data_dir'] = '/vol/val'
    config['train_annotation_file'] = '/vol/train_annotations.json'
    config['val_annotation_file'] = '/vol/val_annotations.json'
    config['checkpoint_dir'] = '/vol/checkpoints'
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {config['device']}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = get_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model_type']}")
    print(f"Total parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(
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
    print(f"Best model saved to: /vol/checkpoints/best_model.pth")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print("="*60)

    return {
        "best_val_loss": trainer.best_val_loss,
        "train_losses": trainer.train_losses,
        "val_losses": trainer.val_losses,
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def download_checkpoint(checkpoint_name: str = "best_model.pth"):
    """
    Download a checkpoint from Modal volume

    Args:
        checkpoint_name: Name of checkpoint file to download
    """
    from pathlib import Path

    checkpoint_path = Path(f"/vol/checkpoints/{checkpoint_name}")

    if not checkpoint_path.exists():
        print(f"Checkpoint {checkpoint_name} not found in volume")
        return None

    with open(checkpoint_path, "rb") as f:
        checkpoint_data = f.read()

    return checkpoint_data


@app.local_entrypoint()
def main(
    download_data: bool = False,
    config: str = "configs/config.yaml",
):
    """
    Main entry point for Modal training

    Usage:
        modal run modal_train.py --download-data
        modal run modal_train.py --config configs/custom_config.yaml
    """
    # Upload source code to Modal
    print("Uploading code to Modal...")

    # Train model
    result = train_model.remote(
        config_path=config,
        download_data=download_data,
    )

    print("\n" + "="*60)
    print("Training Results:")
    print("="*60)
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    print(f"Final train loss: {result['train_losses'][-1]:.4f}")
    print(f"Final val loss: {result['val_losses'][-1]:.4f}")
    print("\nCheckpoints saved to Modal volume 'mabe-data'")
    print("To download: modal run modal_train.py::download_checkpoint")


if __name__ == "__main__":
    main()