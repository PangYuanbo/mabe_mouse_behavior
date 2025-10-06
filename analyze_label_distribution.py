import pandas as pd
import numpy as np
from pathlib import Path
import os

# Path to annotation directory
annotation_dir = Path('data/kaggle/train_annotation')

# Collect all annotation files
all_files = []
for task_folder in annotation_dir.iterdir():
    if task_folder.is_dir():
        for anno_file in task_folder.glob('*.parquet'):
            all_files.append(anno_file)

print(f"Total annotation files found: {len(all_files)}")

# Load first file to understand structure
sample_file = all_files[0]
sample_df = pd.read_parquet(sample_file)
print(f"\nSample file: {sample_file.name}")
print(f"Columns: {sample_df.columns.tolist()}")
print(f"Shape: {sample_df.shape}")
print(f"\nFirst few rows:")
print(sample_df.head(10))

# Analyze all behavior columns
behavior_cols = [col for col in sample_df.columns if col != 'frame_number']
print(f"\nBehavior columns: {behavior_cols}")

# Collect statistics across all files
total_stats = {col: {'0': 0, '1': 0, 'total': 0} for col in behavior_cols}

print("\nAnalyzing all files...")
for i, file in enumerate(all_files):
    if i % 50 == 0:
        print(f"Processing {i}/{len(all_files)}...")

    df = pd.read_parquet(file)

    for col in behavior_cols:
        if col in df.columns:
            counts = df[col].value_counts()
            total_stats[col]['0'] += counts.get(0, 0)
            total_stats[col]['1'] += counts.get(1, 0)
            total_stats[col]['total'] += len(df)

print("\n" + "="*80)
print("LABEL DISTRIBUTION ANALYSIS")
print("="*80)

for col in behavior_cols:
    stats = total_stats[col]
    total = stats['total']
    zeros = stats['0']
    ones = stats['1']

    if total > 0:
        zero_pct = (zeros / total) * 100
        one_pct = (ones / total) * 100

        print(f"\n{col}:")
        print(f"  Total frames: {total:,}")
        print(f"  Label 0 (no behavior): {zeros:,} ({zero_pct:.2f}%)")
        print(f"  Label 1 (behavior present): {ones:,} ({one_pct:.2f}%)")
        print(f"  Class imbalance ratio (0:1): {zeros/ones if ones > 0 else float('inf'):.2f}:1")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Calculate overall statistics
for col in behavior_cols:
    stats = total_stats[col]
    if stats['total'] > 0:
        ratio = stats['0'] / stats['1'] if stats['1'] > 0 else float('inf')
        print(f"{col:30s} - Imbalance: {ratio:6.1f}:1 ({stats['1']/stats['total']*100:5.2f}% positive)")
