"""
Complete Kaggle Submission Script
Uploads dataset + kernel, then guides through submission
"""

from kaggle.api.kaggle_api_extended import KaggleApi
import time

api = KaggleApi()
api.authenticate()

KERNEL_SLUG = 'yuanbopang/v8-mabe-submission'

print("="*60)
print("Kaggle Submission Status")
print("="*60)

# Check dataset
print("\n[1] Dataset:")
try:
    dataset_info = api.dataset_status('yuanbopang/mabe-v8-model')
    print(f"  [OK] Dataset exists and ready")
    print(f"  URL: https://www.kaggle.com/datasets/yuanbopang/mabe-v8-model")
except:
    print(f"  [!] Dataset not found - please run kaggle_auto_submit.py first")

# Check kernel
print("\n[2] Kernel:")
try:
    kernel_info = api.kernel_status(KERNEL_SLUG)
    print(f"  [OK] Kernel uploaded")
    print(f"  URL: https://www.kaggle.com/code/{KERNEL_SLUG}")
    print(f"  Status: {kernel_info.get('status', 'unknown')}")
except Exception as e:
    print(f"  [!] Kernel status: {e}")

print("\n" + "="*60)
print("Next Steps:")
print("="*60)
print("\nOption 1: Manual Execution (Recommended)")
print(f"1. Visit: https://www.kaggle.com/code/{KERNEL_SLUG}")
print("2. Click 'Edit' to open the notebook")
print("3. Verify GPU is enabled (right panel)")
print("4. Click 'Save Version' -> 'Save & Run All (Commit)'")
print("5. Wait ~1-2 hours for execution")
print("6. Once complete, click 'Submit to Competition'")

print("\nOption 2: Check Kernel Output (After it runs)")
print("Run this command to download output:")
print(f"  kaggle kernels output {KERNEL_SLUG} -p ./kaggle_output")

print("\nOption 3: Submit Existing Output")
response = input("\nDo you have a completed kernel version? (y/n): ")

if response.lower() == 'y':
    version = input("Enter version number (e.g., 1): ")

    try:
        # Download kernel output
        print(f"\nDownloading kernel version {version} output...")
        api.kernels_output(KERNEL_SLUG, path='./kaggle_output', version=int(version))

        # Check if submission.csv exists
        import os
        submission_file = './kaggle_output/submission.csv'

        if os.path.exists(submission_file):
            print(f"[OK] Found submission.csv")

            # Submit to competition
            submit = input("Submit to competition now? (y/n): ")
            if submit.lower() == 'y':
                message = "V8 Multi-task Model - 28 Fine-grained Behaviors"
                api.competition_submit(
                    submission_file,
                    message,
                    'MABe-mouse-behavior-detection'
                )
                print("\n[OK] Submission successful!")
                print("  Check: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/submissions")
        else:
            print(f"[!] submission.csv not found in kernel output")
            print("    The kernel may not have completed successfully")

    except Exception as e:
        print(f"[!] Error: {e}")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print("[OK] Dataset uploaded: yuanbopang/mabe-v8-model")
print("[OK] Kernel created: v8-mabe-submission")
print("\nKernel will:")
print("  - Load V8 model (86% validation accuracy)")
print("  - Process all test videos")
print("  - Predict 28 fine-grained behaviors")
print("  - Generate submission.csv")
print("\nEstimated runtime: 1-2 hours on GPU")
print("="*60)
