"""
Detailed per-class metrics for V8
Shows what the model is actually predicting
"""

import numpy as np
from typing import Dict, List
from collections import defaultdict


def compute_per_class_metrics(
    action_preds: np.ndarray,
    action_labels: np.ndarray,
    num_classes: int = 28
) -> Dict:
    """
    Compute detailed per-class metrics

    Args:
        action_preds: [N] predicted action IDs
        action_labels: [N] ground truth action IDs
        num_classes: number of action classes

    Returns:
        Dict with per-class metrics
    """
    metrics = {}

    # Overall counts
    total_samples = len(action_labels)

    for class_id in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((action_preds == class_id) & (action_labels == class_id))
        fp = np.sum((action_preds == class_id) & (action_labels != class_id))
        fn = np.sum((action_preds != class_id) & (action_labels == class_id))
        tn = np.sum((action_preds != class_id) & (action_labels != class_id))

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = tp + fn  # Ground truth count

        # Prediction count (how many times model predicted this class)
        pred_count = tp + fp

        metrics[class_id] = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(support),
            'pred_count': int(pred_count),
            'accuracy': (tp + tn) / total_samples if total_samples > 0 else 0.0
        }

    return metrics


def compute_interval_per_class_f1(
    pred_intervals: List[Dict],
    gt_intervals: List[Dict],
    num_classes: int = 28,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Compute Kaggle-style interval F1 per class

    Args:
        pred_intervals: List of predicted intervals
        gt_intervals: List of ground truth intervals
        num_classes: number of action classes
        iou_threshold: IoU threshold for matching

    Returns:
        Dict with per-class interval metrics
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from versions.v8_fine_grained.submission_utils import compute_iou

    # Group intervals by action
    pred_by_action = defaultdict(list)
    gt_by_action = defaultdict(list)

    for interval in pred_intervals:
        action_id = interval.get('action_id', interval.get('action'))
        pred_by_action[action_id].append(interval)

    for interval in gt_intervals:
        action_id = interval.get('action_id', interval.get('action'))
        gt_by_action[action_id].append(interval)

    # Compute metrics per class
    metrics = {}

    for class_id in range(num_classes):
        pred_class = pred_by_action.get(class_id, [])
        gt_class = gt_by_action.get(class_id, [])

        if len(pred_class) == 0 and len(gt_class) == 0:
            metrics[class_id] = {
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'support': 0,
                'pred_count': 0
            }
            continue

        # Match predictions to ground truth
        matched_gt = set()
        tp = 0

        for pred in pred_class:
            best_iou = 0
            best_gt_idx = -1

            pred_agent = pred.get('agent_id')
            pred_target = pred.get('target_id')

            for gt_idx, gt in enumerate(gt_class):
                if gt_idx in matched_gt:
                    continue

                gt_agent = gt.get('agent_id')
                gt_target = gt.get('target_id')

                # Check agent/target match
                if pred_agent == gt_agent and pred_target == gt_target:
                    iou = compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)

        fp = len(pred_class) - tp
        fn = len(gt_class) - len(matched_gt)

        precision = tp / len(pred_class) if len(pred_class) > 0 else 0.0
        recall = tp / len(gt_class) if len(gt_class) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'support': len(gt_class),
            'pred_count': len(pred_class)
        }

    return metrics


def print_detailed_metrics(
    frame_metrics: Dict,
    interval_metrics: Dict,
    action_names: Dict,
    top_k: int = 10
):
    """
    Print detailed metrics in a readable format

    Args:
        frame_metrics: Per-class frame-level metrics
        interval_metrics: Per-class interval-level metrics
        action_names: Mapping from class_id to action name
        top_k: Show top K classes by support
    """
    print("\n" + "="*80)
    print("DETAILED PER-CLASS METRICS")
    print("="*80)

    # Sort by ground truth support
    sorted_classes = sorted(
        frame_metrics.items(),
        key=lambda x: x[1]['support'],
        reverse=True
    )

    # Header
    print(f"\n{'Class':<20} {'Support':>8} {'Pred':>8} | {'Frame Acc':>9} {'F1':>7} | {'Interval F1':>11} {'Prec':>7} {'Rec':>7}")
    print("-"*80)

    # Show all classes with support > 0 (exclude background for now)
    shown_count = 0
    for class_id, metrics in sorted_classes:
        # Skip background temporarily, will show it separately
        if class_id == 0:
            continue

        # Only show classes that appear in data or predictions
        if metrics['support'] == 0 and metrics['pred_count'] == 0:
            continue

        action_name = action_names.get(class_id, f'class_{class_id}')
        shown_count += 1

        # Stop at top_k if specified
        if top_k is not None and shown_count > top_k:
            break

        # Frame metrics
        support = metrics['support']
        pred_count = metrics['pred_count']
        frame_acc = metrics['accuracy']
        frame_f1 = metrics['f1']

        # Interval metrics
        if class_id in interval_metrics:
            int_metrics = interval_metrics[class_id]
            int_f1 = int_metrics['f1']
            int_prec = int_metrics['precision']
            int_rec = int_metrics['recall']
        else:
            int_f1 = int_prec = int_rec = 0.0

        print(f"{action_name:<20} {support:>8} {pred_count:>8} | "
              f"{frame_acc:>9.4f} {frame_f1:>7.4f} | "
              f"{int_f1:>11.4f} {int_prec:>7.4f} {int_rec:>7.4f}")

    # Show background class separately
    if 0 in frame_metrics:
        print("\n" + "-"*80)
        metrics = frame_metrics[0]
        print(f"{'background':<20} {metrics['support']:>8} {metrics['pred_count']:>8} | "
              f"{metrics['accuracy']:>9.4f} {metrics['f1']:>7.4f} | "
              f"{'N/A':>11} {'N/A':>7} {'N/A':>7}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    # Average metrics (excluding background)
    non_bg_classes = [cid for cid in frame_metrics.keys() if cid != 0]

    if non_bg_classes:
        avg_frame_f1 = np.mean([frame_metrics[cid]['f1'] for cid in non_bg_classes])
        avg_interval_f1 = np.mean([interval_metrics[cid]['f1'] for cid in non_bg_classes if cid in interval_metrics])

        # Weighted by support
        total_support = sum(frame_metrics[cid]['support'] for cid in non_bg_classes)
        if total_support > 0:
            weighted_frame_f1 = sum(
                frame_metrics[cid]['f1'] * frame_metrics[cid]['support']
                for cid in non_bg_classes
            ) / total_support
        else:
            weighted_frame_f1 = 0.0

        print(f"\nMacro-average Frame F1:    {avg_frame_f1:.4f}")
        print(f"Weighted Frame F1:         {weighted_frame_f1:.4f}")
        print(f"Macro-average Interval F1: {avg_interval_f1:.4f}")

    # Class distribution
    classes_with_support = [c for c in non_bg_classes if frame_metrics[c]['support'] > 0]
    classes_predicted = [c for c in non_bg_classes if frame_metrics[c]['pred_count'] > 0]
    classes_missing = [c for c in non_bg_classes if frame_metrics[c]['support'] == 0]

    print(f"\nTotal classes with support: {len(classes_with_support)}")
    print(f"Classes predicted:          {len(classes_predicted)}")
    print(f"Classes missing in data:    {len(classes_missing)}")

    # Show missing classes
    if classes_missing:
        print(f"\nMissing classes (no validation data):")
        for cid in classes_missing:
            action_name = action_names.get(cid, f'class_{cid}')
            print(f"  {cid:>2}: {action_name}")

    # Most confused classes
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)

    # Classes with high FP (model over-predicts)
    over_predicted = sorted(
        [(cid, m) for cid, m in frame_metrics.items() if cid != 0],
        key=lambda x: x[1]['fp'],
        reverse=True
    )[:5]

    print("\nMost Over-predicted (High False Positives):")
    for cid, m in over_predicted:
        action_name = action_names.get(cid, f'class_{cid}')
        print(f"  {action_name:<20} FP: {m['fp']:>6}  (Predicted {m['pred_count']}, True {m['support']})")

    # Classes with high FN (model under-predicts)
    under_predicted = sorted(
        [(cid, m) for cid, m in frame_metrics.items() if cid != 0],
        key=lambda x: x[1]['fn'],
        reverse=True
    )[:5]

    print("\nMost Under-predicted (High False Negatives):")
    for cid, m in under_predicted:
        action_name = action_names.get(cid, f'class_{cid}')
        print(f"  {action_name:<20} FN: {m['fn']:>6}  (Predicted {m['pred_count']}, True {m['support']})")

    print("\n" + "="*80)
