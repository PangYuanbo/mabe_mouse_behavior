"""
Check what actions are in the annotation data
"""

import modal

app = modal.App("check-actions")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "pyarrow")


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def check_actions():
    import pandas as pd
    from pathlib import Path
    from collections import Counter

    annotation_dir = Path("/vol/data/kaggle/train_annotation")

    all_actions = []

    print("Sampling actions from annotation files...")
    for lab_dir in sorted(annotation_dir.iterdir()):
        if lab_dir.is_dir():
            # Sample a few files from each lab
            files = list(lab_dir.glob("*.parquet"))[:3]
            for file in files:
                df = pd.read_parquet(file)
                all_actions.extend(df['action'].tolist())

    action_counts = Counter(all_actions)

    print(f"\n{'='*60}")
    print(f"Action types and counts:")
    print(f"{'='*60}")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {action}: {count}")

    print(f"\nTotal unique actions: {len(action_counts)}")
    print(f"Total action instances: {sum(action_counts.values())}")


@app.local_entrypoint()
def main():
    check_actions.remote()


if __name__ == "__main__":
    main()
