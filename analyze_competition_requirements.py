"""
Analyze what behaviors we actually need to predict based on training data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# Path to data
data_dir = Path('data/kaggle')
annotation_dir = data_dir / 'train_annotation'

print("="*80)
print("ANALYZING COMPETITION REQUIREMENTS")
print("="*80)

# Load train metadata to understand which behaviors are labeled per lab
train_csv = pd.read_csv(data_dir / 'train.csv')

print(f"\nTotal training videos: {len(train_csv)}")
print(f"Unique labs: {train_csv['lab_id'].nunique()}")
print(f"\nLabs: {sorted(train_csv['lab_id'].unique())}")

# Analyze behaviors_labeled column
print("\n" + "="*80)
print("BEHAVIORS LABELED PER LAB")
print("="*80)

behaviors_by_lab = {}
for lab_id in sorted(train_csv['lab_id'].unique()):
    lab_df = train_csv[train_csv['lab_id'] == lab_id]

    # Get behaviors_labeled (may be NaN for some)
    behaviors_labeled = []
    for idx, row in lab_df.iterrows():
        if pd.notna(row['behaviors_labeled']):
            # Behaviors are comma-separated
            behaviors = [b.strip() for b in str(row['behaviors_labeled']).split(',')]
            behaviors_labeled.extend(behaviors)

    behaviors_by_lab[lab_id] = sorted(set(behaviors_labeled))

    if behaviors_labeled:
        print(f"\n{lab_id}:")
        print(f"  Videos: {len(lab_df)}")
        print(f"  Behaviors annotated: {sorted(set(behaviors_labeled))}")

# Now analyze actual annotations to see what actions appear
print("\n" + "="*80)
print("ACTUAL ACTIONS IN ANNOTATIONS")
print("="*80)

actions_by_lab = defaultdict(set)
action_counts = defaultdict(int)

# Scan all annotation files
all_annotation_files = list(annotation_dir.rglob('*.parquet'))
print(f"\nTotal annotation files: {len(all_annotation_files)}")

for anno_file in all_annotation_files:
    lab_id = anno_file.parent.name
    df = pd.read_parquet(anno_file)

    for action in df['action'].unique():
        actions_by_lab[lab_id].add(action)
        action_counts[action] += len(df[df['action'] == action])

print(f"\nActions found by lab:")
for lab_id in sorted(actions_by_lab.keys()):
    print(f"\n{lab_id} ({len(actions_by_lab[lab_id])} actions):")
    print(f"  {sorted(actions_by_lab[lab_id])}")

# All unique actions
all_actions = set()
for actions in actions_by_lab.values():
    all_actions.update(actions)

print("\n" + "="*80)
print("ALL UNIQUE ACTIONS IN TRAINING DATA")
print("="*80)
print(f"\nTotal unique actions: {len(all_actions)}")
print(f"\nActions (sorted by frequency):")

sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
for action, count in sorted_actions:
    print(f"  {action:30s}: {count:6,} intervals")

# Check if there's a pattern - social vs non-social
print("\n" + "="*80)
print("CATEGORIZATION ANALYSIS")
print("="*80)

# Common social behaviors
social_keywords = ['sniff', 'mount', 'attack', 'chase', 'approach', 'follow',
                   'groom', 'huddle', 'dominance', 'escape', 'avoid', 'defend',
                   'flinch', 'intromit', 'ejaculate']

social_actions = []
non_social_actions = []

for action in sorted(all_actions):
    is_social = any(keyword in action.lower() for keyword in social_keywords)
    if is_social:
        social_actions.append(action)
    else:
        non_social_actions.append(action)

print(f"\nPotential SOCIAL behaviors ({len(social_actions)}):")
for action in sorted(social_actions):
    print(f"  - {action}")

print(f"\nPotential NON-SOCIAL behaviors ({len(non_social_actions)}):")
for action in sorted(non_social_actions):
    print(f"  - {action}")

# Key insight
print("\n" + "="*80)
print("COMPETITION REQUIREMENTS SUMMARY")
print("="*80)
print(f"""
Based on the competition description and data analysis:

1. Competition says: "identify over 30 different social and non-social behaviors"
2. Training data contains: {len(all_actions)} unique action types
3. Social behaviors: ~{len(social_actions)} actions
4. Non-social behaviors: ~{len(non_social_actions)} actions

CRITICAL INSIGHT:
The competition explicitly mentions "social AND non-social behaviors".
This means we should NOT map non-social behaviors to background!

Current V8 mapping maps these to background (ID 0):
  - rear, selfgroom, rest, dig, climb
  - exploreobject, biteobject, genitalgroom
  - dominancemount, submit, tussle

These are {len(non_social_actions)} behaviors that might need their own classes!

EVALUATION METHOD:
"F scores are averaged across each lab, each video, and score only the
specific behaviors and mice that were annotated for a specific video."

This means:
- Only behaviors that were annotated in a video are evaluated
- Different labs annotated different behaviors
- We need to predict ALL behaviors that appear in annotations
- NOT just social behaviors!

RECOMMENDATION:
Create a mapping that includes ALL {len(all_actions)} behaviors as separate classes,
not just the {len(social_actions)} social ones.
""")
