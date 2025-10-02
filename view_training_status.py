"""
View H100 training status from checkpoints
"""
import modal

app = modal.App("view-training-status")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

# Need torch in the image to load checkpoints
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.8.0")

@app.function(volumes={"/vol": volume}, image=image, timeout=60)
def view_status():
    """View training status"""
    from pathlib import Path
    import torch
    from datetime import datetime

    print("="*60)
    print("H100 Training Status")
    print("="*60)

    checkpoint_dir = Path("/vol/checkpoints/h100")

    if not checkpoint_dir.exists():
        print("❌ No checkpoint directory")
        return None

    # Find latest checkpoint
    latest_ckpt = checkpoint_dir / "latest_checkpoint.pth"
    best_ckpt = checkpoint_dir / "best_model.pth"

    if not latest_ckpt.exists():
        print("❌ No latest checkpoint")
        return None

    # Load latest checkpoint
    print("\n📦 Loading latest_checkpoint.pth...")
    checkpoint = torch.load(latest_ckpt, map_location='cpu')

    # Display info
    print(f"\n{'='*60}")
    print("Training Progress:")
    print(f"{'='*60}")

    epoch = checkpoint.get('epoch', 'N/A')
    train_loss = checkpoint.get('train_loss', 'N/A')
    val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 'N/A'))
    val_f1 = checkpoint.get('val_f1', checkpoint.get('best_val_f1', 'N/A'))
    val_acc = checkpoint.get('val_acc', 'N/A')

    print(f"  Current Epoch: {epoch}")
    print(f"  Train Loss: {train_loss:.4f}" if isinstance(train_loss, float) else f"  Train Loss: {train_loss}")
    print(f"  Val Loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"  Val Loss: {val_loss}")
    print(f"  Val F1 Macro: {val_f1:.4f}" if isinstance(val_f1, float) else f"  Val F1: {val_f1}")
    print(f"  Val Accuracy: {val_acc:.4f}" if isinstance(val_acc, float) else f"  Val Accuracy: {val_acc}")

    # Best metrics
    print(f"\n{'='*60}")
    print("Best Metrics So Far:")
    print(f"{'='*60}")
    best_val_loss = checkpoint.get('best_val_loss', 'N/A')
    best_val_f1 = checkpoint.get('best_val_f1', 'N/A')

    print(f"  Best Val Loss: {best_val_loss:.4f}" if isinstance(best_val_loss, float) else f"  Best Val Loss: {best_val_loss}")
    print(f"  Best Val F1: {best_val_f1:.4f}" if isinstance(best_val_f1, float) else f"  Best Val F1: {best_val_f1}")

    # File info
    latest_size = latest_ckpt.stat().st_size / 1024 / 1024
    latest_time = datetime.fromtimestamp(latest_ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print("Checkpoint Info:")
    print(f"{'='*60}")
    print(f"  File size: {latest_size:.1f} MB")
    print(f"  Last updated: {latest_time}")

    # Check best model
    if best_ckpt.exists():
        best_checkpoint = torch.load(best_ckpt, map_location='cpu')
        best_epoch = best_checkpoint.get('epoch', 'N/A')
        best_f1_val = best_checkpoint.get('best_val_f1', best_checkpoint.get('val_f1', 'N/A'))
        print(f"\n  Best model saved at epoch: {best_epoch}")
        print(f"  Best F1 score: {best_f1_val:.4f}" if isinstance(best_f1_val, float) else f"  Best F1: {best_f1_val}")

    print("="*60)

    return {
        'epoch': epoch,
        'val_f1': val_f1 if isinstance(val_f1, float) else 0,
        'val_loss': val_loss if isinstance(val_loss, float) else 0,
        'best_val_f1': best_val_f1 if isinstance(best_val_f1, float) else 0
    }


@app.local_entrypoint()
def main():
    """Main entry point"""
    result = view_status.remote()

    if result:
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        print(f"  正在训练 Epoch {result['epoch']}")
        print(f"  当前 F1: {result['val_f1']:.4f}")
        print(f"  最佳 F1: {result['best_val_f1']:.4f}")
        print("="*60)

        # Estimate progress
        if isinstance(result['epoch'], int):
            progress = result['epoch'] / 100 * 100
            remaining = 100 - result['epoch']
            print(f"\n进度: {progress:.1f}% ({result['epoch']}/100 epochs)")
            print(f"剩余: ~{remaining} epochs")

            # Estimate time
            # Assuming ~50s per epoch
            time_per_epoch = 50  # seconds
            remaining_time = remaining * time_per_epoch / 60  # minutes
            print(f"预计剩余时间: ~{remaining_time:.0f} 分钟")

if __name__ == "__main__":
    main()
