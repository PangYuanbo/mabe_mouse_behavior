"""
Test detailed per-class metrics
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.detailed_metrics import compute_per_class_metrics, print_detailed_metrics
from versions.v8_fine_grained.action_mapping import ID_TO_ACTION

# Create dummy data
np.random.seed(42)
n_samples = 10000

# Simulate predictions (biased towards certain classes)
action_preds = np.random.choice([0, 1, 2, 12, 13], size=n_samples, p=[0.7, 0.1, 0.05, 0.1, 0.05])
action_labels = np.random.choice([0, 1, 2, 12, 13], size=n_samples, p=[0.6, 0.15, 0.1, 0.1, 0.05])

# Compute metrics
print("Computing per-class metrics...")
frame_metrics = compute_per_class_metrics(action_preds, action_labels, num_classes=28)

# Create dummy interval metrics (simplified)
interval_metrics = {}
for class_id in range(28):
    if class_id in [0, 1, 2, 12, 13]:
        interval_metrics[class_id] = {
            'precision': np.random.uniform(0.3, 0.9),
            'recall': np.random.uniform(0.3, 0.9),
            'f1': np.random.uniform(0.3, 0.8),
            'tp': frame_metrics[class_id]['tp'] // 10,
            'fp': frame_metrics[class_id]['fp'] // 10,
            'fn': frame_metrics[class_id]['fn'] // 10,
            'support': frame_metrics[class_id]['support'] // 10,
            'pred_count': frame_metrics[class_id]['pred_count'] // 10
        }
    else:
        interval_metrics[class_id] = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'support': 0,
            'pred_count': 0
        }

# Print results
print_detailed_metrics(
    frame_metrics=frame_metrics,
    interval_metrics=interval_metrics,
    action_names=ID_TO_ACTION,
    top_k=10
)

print("\n[OK] Detailed metrics test passed!")
