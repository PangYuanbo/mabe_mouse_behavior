"""
List all checkpoints in Modal volume
"""

import modal

app = modal.App("list-checkpoints")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0")


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def list_checkpoints():
    """List all checkpoints and their metadata"""
    from pathlib import Path
    import os

    checkpoint_dir = Path("/vol/checkpoints/kaggle")

    if not checkpoint_dir.exists():
        print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
        return

    print("="*60)
    print("Available Checkpoints")
    print("="*60)

    checkpoints = sorted(checkpoint_dir.glob("*.pth"))

    if not checkpoints:
        print("\nNo checkpoints found")
        return

    for ckpt_path in checkpoints:
        file_size = os.path.getsize(ckpt_path)
        file_size_mb = file_size / (1024 * 1024)
        mtime = os.path.getmtime(ckpt_path)

        print(f"\n{ckpt_path.name}:")
        print(f"  Size: {file_size_mb:.2f} MB")
        print(f"  Modified: {mtime}")

        # Try to load and inspect checkpoint
        try:
            import torch
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                print(f"  Keys: {list(checkpoint.keys())}")

                if 'epoch' in checkpoint:
                    print(f"  Epoch: {checkpoint['epoch']}")

                if 'best_val_loss' in checkpoint:
                    print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")

                if 'best_val_f1' in checkpoint:
                    print(f"  Best Val F1: {checkpoint['best_val_f1']:.4f}")

                if 'val_losses' in checkpoint:
                    print(f"  Total epochs trained: {len(checkpoint['val_losses'])}")

                if 'val_metrics' in checkpoint and len(checkpoint['val_metrics']) > 0:
                    latest_metrics = checkpoint['val_metrics'][-1]
                    print(f"  Latest val metrics: {latest_metrics}")

        except Exception as e:
            print(f"  ✗ Could not load checkpoint: {e}")

    print("\n" + "="*60)
    print(f"Total checkpoints: {len(checkpoints)}")
    print("="*60)


@app.local_entrypoint()
def main():
    list_checkpoints.remote()


if __name__ == "__main__":
    main()
