"""
Check H100 training progress
"""
import modal

app = modal.App("check-h100-progress")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

@app.function(volumes={"/vol": volume}, timeout=60)
def check_progress():
    """Check training progress"""
    from pathlib import Path
    import os

    print("="*60)
    print("H100 Training Progress Check")
    print("="*60)

    # Check checkpoint directory
    checkpoint_dir = Path("/vol/checkpoints/h100")

    if not checkpoint_dir.exists():
        print("❌ Checkpoint directory doesn't exist yet")
        print("   Training may still be loading data...")
        return None

    # List all checkpoints
    checkpoints = sorted(list(checkpoint_dir.glob("*.pth")))

    if not checkpoints:
        print("❌ No checkpoints found yet")
        print("   Training may still be in early stages...")
        return None

    print(f"\n✓ Found {len(checkpoints)} checkpoints:")
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / 1024 / 1024
        mtime = ckpt.stat().st_mtime
        from datetime import datetime
        time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  - {ckpt.name}: {size_mb:.1f} MB (modified: {time_str})")

    # Check for training log
    log_files = list(checkpoint_dir.glob("*.log"))
    if log_files:
        print(f"\n✓ Found {len(log_files)} log files")
        latest_log = sorted(log_files)[-1]
        print(f"\nLast 20 lines of {latest_log.name}:")
        print("-" * 60)
        with open(latest_log) as f:
            lines = f.readlines()
            for line in lines[-20:]:
                print(line.rstrip())

    # Check latest checkpoint
    if checkpoints:
        latest_ckpt = sorted(checkpoints)[-1]
        print(f"\n📦 Latest checkpoint: {latest_ckpt.name}")

        # Try to load and inspect
        try:
            import torch
            checkpoint = torch.load(latest_ckpt, map_location='cpu')
            print(f"\nCheckpoint contents:")
            for key in checkpoint.keys():
                print(f"  - {key}")

            if 'epoch' in checkpoint:
                print(f"\n📊 Training Status:")
                print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
                print(f"  Best Val F1: {checkpoint.get('best_val_f1', 'N/A'):.4f}")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")

    print("="*60)

    return {
        "num_checkpoints": len(checkpoints),
        "latest_checkpoint": checkpoints[-1].name if checkpoints else None
    }


@app.local_entrypoint()
def main():
    """Main entry point"""
    result = check_progress.remote()

    if result:
        print(f"\n✅ Training is running!")
        print(f"   Checkpoints: {result['num_checkpoints']}")
        print(f"   Latest: {result['latest_checkpoint']}")
    else:
        print(f"\n⏳ Training is still in early stages...")
        print(f"   Please wait a few minutes and check again.")

if __name__ == "__main__":
    main()
