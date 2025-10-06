"""
Check which behaviors are missing from the training data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from versions.v8_fine_grained.action_mapping import ACTION_TO_ID, ID_TO_ACTION

# Path to annotation directory
annotation_dir = Path('data/kaggle/train_annotation')

# Collect all annotation files
all_files = []
for task_folder in annotation_dir.iterdir():
    if task_folder.is_dir():
        for anno_file in task_folder.glob('*.parquet'):
            all_files.append(anno_file)

print(f"Total annotation files: {len(all_files)}")
print("="*80)

# Collect all unique actions in the data
actions_in_data = set()

print("Scanning all annotation files for action types...")
for i, file in enumerate(all_files):
    if i % 100 == 0:
        print(f"Processing {i}/{len(all_files)}...")

    try:
        df = pd.read_parquet(file)
        actions_in_data.update(df['action'].unique())
    except Exception as e:
        print(f"Error processing {file}: {e}")

print(f"\nTotal unique actions found in data: {len(actions_in_data)}")
print("\n" + "="*80)
print("ACTIONS FOUND IN DATA")
print("="*80)

# Sort alphabetically
sorted_actions = sorted(actions_in_data)
for action in sorted_actions:
    action_id = ACTION_TO_ID.get(action, 0)
    mapped_name = ID_TO_ACTION.get(action_id, 'background')
    print(f"  {action:30s} -> ID {action_id:2d} ({mapped_name})")

print("\n" + "="*80)
print("ACTIONS DEFINED IN ACTION_MAPPING BUT NOT IN DATA")
print("="*80)

# Get all actions defined in ACTION_TO_ID (excluding background)
defined_behaviors = set()
for action_name, action_id in ACTION_TO_ID.items():
    if action_id > 0:  # Exclude background
        defined_behaviors.add(action_name)

# Check which defined behaviors are missing from data
missing_behaviors = defined_behaviors - actions_in_data

if missing_behaviors:
    print(f"\nFound {len(missing_behaviors)} behaviors defined in mapping but MISSING from data:\n")
    for action in sorted(missing_behaviors):
        action_id = ACTION_TO_ID.get(action)
        print(f"  {action:30s} (ID {action_id})")
else:
    print("\nAll defined behaviors are present in the data!")

print("\n" + "="*80)
print("ACTIONS IN DATA BUT NOT MAPPED TO BEHAVIOR CLASS")
print("="*80)

# Check which actions in data are not mapped (mapped to background)
unmapped_actions = []
for action in sorted(actions_in_data):
    action_id = ACTION_TO_ID.get(action, 0)
    if action_id == 0 and action != 'background':
        unmapped_actions.append(action)

if unmapped_actions:
    print(f"\nFound {len(unmapped_actions)} actions mapped to background (ID 0):\n")
    for action in unmapped_actions:
        print(f"  {action}")
else:
    print("\nNo unmapped actions found.")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Actions in data:              {len(actions_in_data)}")
print(f"Defined behavior classes:     {len(defined_behaviors)}")
print(f"Missing from data:            {len(missing_behaviors)}")
print(f"Mapped to background:         {len(unmapped_actions)}")
print(f"Active behavior classes:      {len([a for a in actions_in_data if ACTION_TO_ID.get(a, 0) > 0])}")
