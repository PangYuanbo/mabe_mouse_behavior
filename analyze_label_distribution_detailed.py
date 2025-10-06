"""
Analyze label distribution in MABe mouse behavior training data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from versions.v8_fine_grained.action_mapping import ACTION_TO_ID, ID_TO_ACTION, get_category

# Path to annotation directory
annotation_dir = Path('data/kaggle/train_annotation')

# Collect all annotation files
all_files = []
for task_folder in annotation_dir.iterdir():
    if task_folder.is_dir():
        for anno_file in task_folder.glob('*.parquet'):
            all_files.append(anno_file)

print(f"Total annotation files found: {len(all_files)}")
print("="*80)

# Load first file to understand structure
sample_file = all_files[0]
sample_df = pd.read_parquet(sample_file)
print(f"\nSample file: {sample_file.name}")
print(f"Columns: {sample_df.columns.tolist()}")
print(f"Shape: {sample_df.shape}")
print(f"\nFirst 10 rows:")
print(sample_df.head(10))
print("\n" + "="*80)

# Collect statistics on action types
action_stats = {}
frame_label_stats = {action_id: 0 for action_id in range(28)}  # Frame-level stats
total_frames = 0
total_intervals = 0

print("\nAnalyzing all annotation files...")
for i, file in enumerate(all_files):
    if i % 50 == 0:
        print(f"Processing {i}/{len(all_files)}...")

    try:
        df = pd.read_parquet(file)

        # Count intervals (rows in annotation file)
        total_intervals += len(df)

        # Count actions
        for _, row in df.iterrows():
            action = row['action']
            start_frame = row['start_frame']
            stop_frame = row['stop_frame']
            duration = stop_frame - start_frame + 1

            # Count action occurrences
            if action not in action_stats:
                action_stats[action] = {
                    'count': 0,
                    'total_frames': 0,
                    'durations': []
                }

            action_stats[action]['count'] += 1
            action_stats[action]['total_frames'] += duration
            action_stats[action]['durations'].append(duration)

            # Map to action ID for frame-level stats
            action_id = ACTION_TO_ID.get(action, 0)
            frame_label_stats[action_id] += duration

        # Get total frames from this video (max stop_frame)
        if len(df) > 0:
            total_frames += df['stop_frame'].max() + 1

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("\n" + "="*80)
print("INTERVAL-LEVEL STATISTICS")
print("="*80)

# Sort actions by frequency
sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['count'], reverse=True)

print(f"\nTotal intervals: {total_intervals:,}")
print(f"Total unique action types: {len(action_stats)}")
print(f"\nAction frequency (sorted by occurrence count):\n")

for action, stats in sorted_actions:
    count = stats['count']
    total_dur = stats['total_frames']
    avg_dur = np.mean(stats['durations'])
    med_dur = np.median(stats['durations'])

    print(f"{action:25s} | Count: {count:6,} ({count/total_intervals*100:5.2f}%) | "
          f"Avg dur: {avg_dur:6.1f} frames | Med dur: {med_dur:5.0f} frames")

print("\n" + "="*80)
print("FRAME-LEVEL STATISTICS (What the model sees)")
print("="*80)

# Calculate background frames
labeled_frames = sum(frame_label_stats.values())
background_frames = total_frames - labeled_frames

print(f"\nTotal frames: {total_frames:,}")
print(f"Background frames: {background_frames:,} ({background_frames/total_frames*100:.2f}%)")
print(f"Labeled frames: {labeled_frames:,} ({labeled_frames/total_frames*100:.2f}%)")

# Sort by frame count
sorted_frame_stats = sorted(
    [(action_id, count) for action_id, count in frame_label_stats.items() if count > 0],
    key=lambda x: x[1],
    reverse=True
)

print(f"\nAction distribution by frame count (excluding background):\n")
print(f"{'Action ID':<12} {'Action Name':<25} {'Frame Count':<15} {'% of Total':<12} {'% of Labeled':<12} {'Category'}")
print("-" * 110)

for action_id, count in sorted_frame_stats:
    action_name = ID_TO_ACTION.get(action_id, 'unknown')
    category = get_category(action_id)
    pct_total = count / total_frames * 100
    pct_labeled = count / labeled_frames * 100 if labeled_frames > 0 else 0

    print(f"{action_id:<12} {action_name:<25} {count:<15,} {pct_total:>6.3f}%      {pct_labeled:>6.2f}%      {category}")

print("\n" + "="*80)
print("CLASS IMBALANCE ANALYSIS")
print("="*80)

# Calculate imbalance ratios (relative to each behavior)
print(f"\nBackground vs Behavior ratios:\n")
for action_id, count in sorted_frame_stats[:10]:  # Top 10 most common
    action_name = ID_TO_ACTION.get(action_id, 'unknown')
    ratio = background_frames / count if count > 0 else float('inf')
    print(f"{action_name:25s}: {ratio:8.1f}:1 (background:behavior)")

# Category statistics
print(f"\n" + "="*80)
print("CATEGORY STATISTICS")
print("="*80)

from versions.v8_fine_grained.action_mapping import ACTION_CATEGORIES

category_frames = {cat: 0 for cat in ACTION_CATEGORIES.keys()}
for action_id, count in frame_label_stats.items():
    if action_id == 0:
        continue
    category = get_category(action_id)
    if category in category_frames:
        category_frames[category] += count

print(f"\nFrames per category:")
for cat, count in sorted(category_frames.items(), key=lambda x: x[1], reverse=True):
    pct = count / total_frames * 100
    print(f"{cat:15s}: {count:12,} frames ({pct:5.2f}%)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"The dataset has severe class imbalance:")
print(f"  • Background dominates with {background_frames/total_frames*100:.1f}% of frames")
print(f"  • {len(sorted_frame_stats)} behavior classes share {labeled_frames/total_frames*100:.1f}% of frames")
print(f"  • Most behaviors appear in <1% of frames")
print(f"  • Imbalance ratios range from hundreds to thousands to one")
print(f"\nRecommendations:")
print(f"  1. Use class weights in loss function")
print(f"  2. Use focal loss for hard examples")
print(f"  3. Consider oversampling rare behaviors")
print(f"  4. Use balanced batch sampling")
