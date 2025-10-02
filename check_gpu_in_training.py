"""
Check GPU usage WITHIN the training container
This needs to be integrated into the training script
"""
import modal

app = modal.App("check-training-gpu")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0")
)

@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": volume},
    timeout=120,
)
def check_gpu_with_torch():
    """Check GPU with torch (simulates training environment)"""
    import torch
    import subprocess

    print("="*60)
    print("GPU Check in Training-like Environment")
    print("="*60)

    # Check torch
    print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")

        # Memory info
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\nMemory Allocated: {mem_allocated:.2f} GB")
        print(f"Memory Reserved: {mem_reserved:.2f} GB")
        print(f"Memory Total: {mem_total:.2f} GB")

    # Run nvidia-smi
    print("\n" + "="*60)
    print("nvidia-smi output:")
    print("="*60)
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)

    # Detailed query
    print("\n" + "="*60)
    print("Detailed GPU Stats:")
    print("="*60)
    result = subprocess.run(
        ["nvidia-smi", "dmon", "-c", "1"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    return True


@app.local_entrypoint()
def main():
    """Main"""
    check_gpu_with_torch.remote()

if __name__ == "__main__":
    main()
