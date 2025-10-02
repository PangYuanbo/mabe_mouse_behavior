"""
Download best_model.pth from Modal volume to local
"""

import modal

app = modal.App("download-checkpoint")
volume = modal.Volume.from_name("mabe-data", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def download_checkpoint(checkpoint_name: str = "best_model.pth", checkpoint_dir: str = None):
    """Download checkpoint from Modal volume"""
    from pathlib import Path

    # Try multiple paths
    if checkpoint_dir:
        checkpoint_paths = [Path(checkpoint_dir) / checkpoint_name]
    else:
        checkpoint_paths = [
            Path("/vol/checkpoints/kaggle") / checkpoint_name,
            Path("/vol/checkpoints/h100") / checkpoint_name,
            Path("/vol/checkpoints/5090") / checkpoint_name,
            Path("/vol/checkpoints") / checkpoint_name,
        ]

    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            print(f"✓ Found checkpoint: {checkpoint_path}")
            print(f"  Size: {checkpoint_path.stat().st_size / 1024**2:.1f} MB")

            with open(checkpoint_path, "rb") as f:
                return f.read()

    print(f"✗ Checkpoint not found in any location:")
    for path in checkpoint_paths:
        print(f"  - {path}")
    return None


@app.local_entrypoint()
def main(checkpoint: str = "best_model.pth", output: str = "best_model.pth", dir: str = None):
    """
    Download checkpoint from Modal

    Args:
        checkpoint: Checkpoint filename (default: best_model.pth)
        output: Local output filename (default: best_model.pth)
        dir: Specific checkpoint directory (e.g., /vol/checkpoints/kaggle)
    """
    print("\n" + "="*60)
    print("Downloading Checkpoint from Modal")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    if dir:
        print(f"Directory: {dir}")
    print(f"Output: {output}")
    print("="*60 + "\n")

    content = download_checkpoint.remote(checkpoint_name=checkpoint, checkpoint_dir=dir)

    if content:
        with open(output, "wb") as f:
            f.write(content)
        print(f"\n✓ Downloaded to: {output}")
        print(f"  Size: {len(content) / 1024**2:.1f} MB")
    else:
        print("\n✗ Download failed")


if __name__ == "__main__":
    main()
