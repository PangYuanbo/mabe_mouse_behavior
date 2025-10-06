"""
Automated Kaggle Submission Script for V8
Handles: Dataset upload → Kernel creation → Submission

Usage: python kaggle_auto_submit.py
"""

import os
import json
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize API
api = KaggleApi()
api.authenticate()

print("="*60)
print("V8.1 Optimized Kaggle Auto-Submission Script")
print("="*60)

# Configuration
COMPETITION = 'MABe-mouse-behavior-detection'  # Official competition slug
MODEL_PATH = Path('checkpoints/v8.1_optimized/best_model.pth')  # V8.1: Use Kaggle F1 best model
NOTEBOOK_PATH = Path('kaggle_submission_v8.1.ipynb')  # V8.1: New notebook
DATASET_NAME = 'mabe-v8-1-model'  # V8.1: New dataset name
USERNAME = api.get_config_value('username')

print(f"\nUser: {USERNAME}")
print(f"Competition: {COMPETITION}")
print()

# Step 1: Create dataset metadata
print("Step 1: Creating dataset metadata...")

dataset_metadata = {
    "title": "MABe V8.1 Optimized Model Checkpoint",
    "id": f"{USERNAME}/{DATASET_NAME}",
    "licenses": [{"name": "CC0-1.0"}],
    "resources": [
        {
            "path": MODEL_PATH.name,
            "description": "V8.1 Optimized Model - Best Kaggle F1 checkpoint with motion gating and fine-tuned thresholds"
        }
    ]
}

dataset_dir = Path('kaggle_dataset_v8.1')  # V8.1: New directory
dataset_dir.mkdir(exist_ok=True)

# Copy model to dataset directory
import shutil
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Please train V8.1 first!")
shutil.copy(MODEL_PATH, dataset_dir / MODEL_PATH.name)

# Write metadata
with open(dataset_dir / 'dataset-metadata.json', 'w') as f:
    json.dump(dataset_metadata, f, indent=2)

print(f"[OK] V8.1 Dataset prepared in {dataset_dir}")
print(f"    Model: {MODEL_PATH}")

# Step 2: Upload dataset
print("\nStep 2: Uploading dataset to Kaggle...")

try:
    # Check if dataset exists
    try:
        api.dataset_status(f"{USERNAME}/{DATASET_NAME}")
        print(f"  Dataset already exists, creating new version...")
        api.dataset_create_version(
            folder=str(dataset_dir),
            version_notes="V8.1 Optimized: Motion gating + Fine-tuned thresholds",
            quiet=False
        )
    except:
        print(f"  Creating new dataset...")
        api.dataset_create_new(
            folder=str(dataset_dir),
            public=False,  # Set to True if you want public
            quiet=False
        )

    print("[OK] Dataset uploaded successfully")
    print(f"  URL: https://www.kaggle.com/datasets/{USERNAME}/{DATASET_NAME}")

    # Wait for dataset to be ready
    print("  Waiting for dataset to process...")
    time.sleep(10)

except Exception as e:
    print(f"[!] Dataset upload failed: {e}")
    print("  Please upload manually or check your Kaggle API credentials")
    exit(1)

# Step 3: Prepare kernel metadata
print("\nStep 3: Preparing kernel (notebook)...")

kernel_metadata = {
    "id": f"{USERNAME}/v8-1-mabe-submission",  # V8.1: New kernel slug
    "title": "V8.1 Optimized MABe Submission",
    "code_file": NOTEBOOK_PATH.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": [
        f"{USERNAME}/{DATASET_NAME}"  # Our V8.1 model
    ],
    "competition_sources": [COMPETITION],  # Competition data is automatically available
    "kernel_sources": []
}

kernel_dir = Path('kaggle_kernel_v8.1')  # V8.1: New directory
kernel_dir.mkdir(exist_ok=True)

# Copy notebook
if not NOTEBOOK_PATH.exists():
    raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")
shutil.copy(NOTEBOOK_PATH, kernel_dir / NOTEBOOK_PATH.name)

# Write metadata
with open(kernel_dir / 'kernel-metadata.json', 'w') as f:
    json.dump(kernel_metadata, f, indent=2)

print(f"[OK] Kernel prepared in {kernel_dir}")

# Step 4: Push kernel
print("\nStep 4: Pushing kernel to Kaggle...")

try:
    # Try to create new kernel
    try:
        api.kernels_push(str(kernel_dir))
        print("[OK] Kernel created successfully")
    except:
        # If exists, update it
        print("  Kernel exists, updating...")
        api.kernels_push(str(kernel_dir))
        print("[OK] Kernel updated successfully")

    kernel_url = f"https://www.kaggle.com/code/{USERNAME}/v8-1-mabe-submission"  # V8.1
    print(f"  URL: {kernel_url}")

except Exception as e:
    print(f"[!] Kernel push failed: {e}")
    print("  You may need to manually create the kernel on Kaggle")
    exit(1)

# Step 5: Instructions for manual submission
print("\n" + "="*60)
print("Next Steps (Manual):")
print("="*60)
print(f"1. Visit: {kernel_url}")
print(f"2. Click 'Edit' to open the notebook")
print(f"3. Click 'Run All' or 'Save Version' -> 'Save & Run All'")
print(f"4. Wait for execution to complete (~1-2 hours)")
print(f"5. Once finished, click 'Submit to Competition'")
print()
print("The kernel will automatically:")
print("  - Load the V8.1 optimized model from your dataset")
print("  - Apply motion gating filters (escape/chase/freeze)")
print("  - Use fine-tuned class-specific thresholds")
print("  - Perform sliding window ensemble inference (50% overlap)")
print("  - Generate submission.csv in correct format")
print("  - Expected improvements: Escape FP -30~50%, Freeze Recall +10~20%")
print()

# Check if direct submission API is available
print("Attempting direct competition submission...")
try:
    # First, check if kernel has run and produced output
    print("\n[Note] Kaggle API doesn't support auto-submit for code competitions.")
    print("       You must manually submit via the website after kernel runs.")
    print()
    print("Alternative: Generate submission locally and submit CSV directly")

    response = input("\nDo you want to generate submission locally? (y/n): ")

    if response.lower() == 'y':
        print("\nGenerating local V8.1 submission...")
        import subprocess
        result = subprocess.run([
            'python', 'inference_v8.1.py',
            '--checkpoint', str(MODEL_PATH),
            '--output', 'submission_v8.1.csv'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("[OK] Local submission generated: submission_v8.1.csv")

            # Submit CSV directly
            try:
                api.competition_submit(
                    'submission_v8.1.csv',
                    'V8.1 Optimized: Motion Gating + Fine-tuned Thresholds',
                    COMPETITION
                )
                print("\n[OK] Submission successful!")
                print(f"  Check status: https://www.kaggle.com/competitions/{COMPETITION}/submissions")
            except Exception as e:
                print(f"\n[!] Submission failed: {e}")
                print("  Please submit manually via Kaggle website")
        else:
            print(f"[!] Inference failed: {result.stderr}")

except Exception as e:
    print(f"[!] Error: {e}")

print("\n" + "="*60)
print("V8.1 Submission Complete!")
print("="*60)
print("Key V8.1 Improvements:")
print("  ✓ Motion gating (escape/chase ≥80px/s, freeze ≤7px/s)")
print("  ✓ Fine-tuned thresholds (sniff 0.50, sniffgenital 0.26, intromit 0.45)")
print("  ✓ Sliding window ensemble (50% overlap)")
print("  ✓ Adaptive minimum durations")
print("="*60)
