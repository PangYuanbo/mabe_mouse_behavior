"""
Monitor H100 GPU usage during training
"""
import modal

app = modal.App("mabe-h100-monitor")

@app.function(gpu="H100", timeout=60)
def check_gpu():
    """Check GPU usage"""
    import subprocess

    # Run nvidia-smi
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True
    )

    gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(",")

    print("="*60)
    print("H100 GPU Status")
    print("="*60)
    print(f"GPU Utilization: {gpu_util.strip()}%")
    print(f"Memory Used: {mem_used.strip()} MB / {mem_total.strip()} MB")
    print(f"Memory Usage: {float(mem_used) / float(mem_total) * 100:.1f}%")
    print(f"Temperature: {temp.strip()}°C")
    print("="*60)

    # Detailed nvidia-smi output
    full_output = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("\nDetailed nvidia-smi output:")
    print(full_output.stdout)

    return {
        "gpu_util": float(gpu_util),
        "mem_used_mb": float(mem_used),
        "mem_total_mb": float(mem_total),
        "mem_usage_pct": float(mem_used) / float(mem_total) * 100,
        "temperature": float(temp)
    }


@app.local_entrypoint()
def main():
    """Monitor H100"""
    stats = check_gpu.remote()

    print("\n" + "="*60)
    print("Summary:")
    print(f"  GPU使用率: {stats['gpu_util']}%")
    print(f"  显存使用: {stats['mem_used_mb']:.0f} MB / {stats['mem_total_mb']:.0f} MB ({stats['mem_usage_pct']:.1f}%)")
    print(f"  温度: {stats['temperature']}°C")

    if stats['gpu_util'] < 50:
        print("\n⚠️  WARNING: GPU利用率较低！可能batch size还能增大")
    elif stats['gpu_util'] > 95:
        print("\n✅ GPU利用率很高，性能充分")
    else:
        print("\n✓ GPU利用率正常")

    if stats['mem_usage_pct'] < 30:
        print("⚠️  WARNING: 显存使用率很低！建议增大batch size")
    elif stats['mem_usage_pct'] > 90:
        print("⚠️  WARNING: 显存接近满载，小心OOM")
    else:
        print("✓ 显存使用率合理")

    print("="*60)


if __name__ == "__main__":
    main()
