"""
Check label distribution in the dataset
"""

import modal

app = modal.App("check-labels")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.8.0",
    "numpy==2.3.3",
    "pandas==2.3.3",
    "pyarrow==21.0.0",
)


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def check_labels():
    import sys
    sys.path.insert(0, "/vol/code")

    import numpy as np
    from collections import Counter

    from src.data.kaggle_dataset import KaggleMABeDataset

    print("Loading training dataset...")
    dataset = KaggleMABeDataset(
        data_dir='/vol/data/kaggle',
        split='train',
        sequence_length=100,
        max_sequences=20,  # Sample 20 videos
        use_feature_engineering=False,
    )

    print(f"Total sequences: {len(dataset)}")

    # Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        _, labels = dataset[i]
        all_labels.extend(labels.numpy().tolist())

    # Count labels
    label_counts = Counter(all_labels)
    total = len(all_labels)

    print(f"\n{'='*60}")
    print("Label Distribution")
    print(f"{'='*60}")
    print(f"Total frames: {total:,}")
    print(f"\nLabel breakdown:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / total * 100
        print(f"  Label {label}: {count:,} frames ({pct:.2f}%)")

    # Calculate class imbalance ratio
    max_count = max(label_counts.values())
    min_count = min(label_counts.values()) if min(label_counts.values()) > 0 else 1
    imbalance_ratio = max_count / min_count

    print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")

    # Suggest class weights
    print(f"\n{'='*60}")
    print("Suggested class weights (inverse frequency):")
    print(f"{'='*60}")
    for label in sorted(label_counts.keys()):
        weight = total / (len(label_counts) * label_counts[label])
        print(f"  Label {label}: {weight:.2f}")


@app.local_entrypoint()
def main():
    check_labels.remote()


if __name__ == "__main__":
    main()
