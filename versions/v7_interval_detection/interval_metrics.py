"""
V7: Evaluation metrics for interval detection
F1 score based on IoU matching
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class IntervalMetrics:
    """
    Evaluation metrics for temporal action detection
    Computes F1 score with IoU-based matching
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """Reset accumulated metrics"""
        self.all_predictions = []
        self.all_targets = []

    def update(self, predictions: List[List[Dict]], targets: List[List[Dict]]):
        """
        Update metrics with batch predictions and targets

        Args:
            predictions: List of predicted intervals per video
            targets: List of ground truth intervals per video
        """
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics

        Returns:
            dict with precision, recall, f1
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_intervals, gt_intervals in zip(self.all_predictions, self.all_targets):
            # Match predictions to ground truth
            matched_gt = set()

            for pred in pred_intervals:
                # Find best matching GT
                best_match = None
                best_iou = 0

                for gt_idx, gt in enumerate(gt_intervals):
                    if gt_idx in matched_gt:
                        continue

                    # Compute IoU
                    iou = self.compute_iou(
                        [pred['start_frame'], pred['end_frame']],
                        [gt['start_frame'], gt['end_frame']]
                    )

                    if iou > best_iou:
                        best_iou = iou
                        best_match = gt_idx

                # Check if match is valid
                if best_match is not None and best_iou >= self.iou_threshold:
                    gt = gt_intervals[best_match]

                    # Check if action, agent, target match
                    if (pred['action_id'] == gt['action_id'] and
                        pred['agent_id'] == gt['agent_id'] and
                        pred['target_id'] == gt['target_id']):

                        true_positives += 1
                        matched_gt.add(best_match)
                    else:
                        false_positives += 1
                else:
                    false_positives += 1

            # Unmatched ground truth = false negatives
            false_negatives += len(gt_intervals) - len(matched_gt)

        # Compute metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
        }

    def compute_per_action(self) -> Dict[str, Dict[str, float]]:
        """Compute metrics per action class"""
        action_names = {0: 'attack', 1: 'avoid', 2: 'chase', 3: 'chaseattack'}

        per_action_metrics = {}

        for action_id, action_name in action_names.items():
            tp = fp = fn = 0

            for pred_intervals, gt_intervals in zip(self.all_predictions, self.all_targets):
                # Filter by action
                pred_action = [p for p in pred_intervals if p['action_id'] == action_id]
                gt_action = [g for g in gt_intervals if g['action_id'] == action_id]

                matched_gt = set()

                for pred in pred_action:
                    best_match = None
                    best_iou = 0

                    for gt_idx, gt in enumerate(gt_action):
                        if gt_idx in matched_gt:
                            continue

                        iou = self.compute_iou(
                            [pred['start_frame'], pred['end_frame']],
                            [gt['start_frame'], gt['end_frame']]
                        )

                        if iou > best_iou:
                            best_iou = iou
                            best_match = gt_idx

                    if best_match is not None and best_iou >= self.iou_threshold:
                        gt = gt_action[best_match]

                        if (pred['agent_id'] == gt['agent_id'] and
                            pred['target_id'] == gt['target_id']):
                            tp += 1
                            matched_gt.add(best_match)
                        else:
                            fp += 1
                    else:
                        fp += 1

                fn += len(gt_action) - len(matched_gt)

            # Compute metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_action_metrics[action_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
            }

        return per_action_metrics

    @staticmethod
    def compute_iou(interval1: List[int], interval2: List[int]) -> float:
        """Compute IoU between two intervals"""
        start1, end1 = interval1
        start2, end2 = interval2

        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)

        if intersection_start >= intersection_end:
            return 0.0

        intersection = intersection_end - intersection_start
        union = (end1 - start1) + (end2 - start2) - intersection

        return intersection / union if union > 0 else 0.0


def evaluate_intervals(
    predictions: List[List[Dict]],
    targets: List[List[Dict]],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Convenience function to evaluate interval predictions

    Args:
        predictions: List of predicted intervals per video
        targets: List of ground truth intervals per video
        iou_threshold: IoU threshold for matching

    Returns:
        dict with overall and per-action metrics
    """
    metrics = IntervalMetrics(iou_threshold=iou_threshold)
    metrics.update(predictions, targets)

    overall = metrics.compute()
    per_action = metrics.compute_per_action()

    return {
        'overall': overall,
        'per_action': per_action,
    }
