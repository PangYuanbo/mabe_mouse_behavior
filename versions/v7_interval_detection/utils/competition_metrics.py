"""
Competition-style interval-based metrics for MABe challenge
Evaluates predicted intervals against ground truth intervals
"""

import numpy as np
from typing import List, Dict, Tuple
from .interval_converter import match_intervals


class IntervalMetrics:
    """
    Compute interval-based precision, recall, F1 for temporal action detection
    Follows MABe competition evaluation protocol
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        match_action: bool = True,
        num_actions: int = 4,
    ):
        """
        Args:
            iou_threshold: Minimum IoU to consider a match
            match_action: If True, require action_id to match for valid detection
            num_actions: Number of action classes (including background as 0)
        """
        self.iou_threshold = iou_threshold
        self.match_action = match_action
        self.num_actions = num_actions

        # Accumulators
        self.reset()

    def reset(self):
        """Reset accumulators"""
        self.total_tp = 0  # True positives
        self.total_fp = 0  # False positives
        self.total_fn = 0  # False negatives

        # Per-action statistics
        self.per_action_tp = {i: 0 for i in range(self.num_actions)}
        self.per_action_fp = {i: 0 for i in range(self.num_actions)}
        self.per_action_fn = {i: 0 for i in range(self.num_actions)}

    def update(
        self,
        pred_intervals: List[List[Dict]],
        gt_intervals: List[List[Dict]],
    ):
        """
        Update metrics with batch predictions

        Args:
            pred_intervals: List of predicted intervals per sample
                Each interval: {'start_frame', 'end_frame', 'action_id', ...}
            gt_intervals: List of ground truth intervals per sample
        """
        batch_size = len(pred_intervals)

        for b in range(batch_size):
            preds = pred_intervals[b]
            gts = gt_intervals[b]

            # Match predictions to ground truth
            matches, unmatched_preds, unmatched_gts = match_intervals(
                preds,
                gts,
                iou_threshold=self.iou_threshold,
                match_action=self.match_action
            )

            # True positives: matched predictions
            tp = len(matches)
            self.total_tp += tp

            # False positives: unmatched predictions
            fp = len(unmatched_preds)
            self.total_fp += fp

            # False negatives: unmatched ground truth
            fn = len(unmatched_gts)
            self.total_fn += fn

            # Per-action statistics
            for pred_idx, gt_idx in matches:
                action_id = gts[gt_idx]['action_id']
                self.per_action_tp[action_id] += 1

            for pred_idx in unmatched_preds:
                action_id = preds[pred_idx]['action_id']
                self.per_action_fp[action_id] += 1

            for gt_idx in unmatched_gts:
                action_id = gts[gt_idx]['action_id']
                self.per_action_fn[action_id] += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute overall metrics

        Returns:
            metrics: Dict with 'precision', 'recall', 'f1'
        """
        if self.total_tp + self.total_fp == 0:
            precision = 0.0
        else:
            precision = self.total_tp / (self.total_tp + self.total_fp)

        if self.total_tp + self.total_fn == 0:
            recall = 0.0
        else:
            recall = self.total_tp / (self.total_tp + self.total_fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': self.total_tp,
            'fp': self.total_fp,
            'fn': self.total_fn,
        }

    def compute_per_action(self) -> Dict[int, Dict[str, float]]:
        """
        Compute per-action metrics

        Returns:
            per_action_metrics: Dict mapping action_id -> metrics dict
        """
        per_action_metrics = {}

        for action_id in range(1, self.num_actions):  # Skip background (0)
            tp = self.per_action_tp[action_id]
            fp = self.per_action_fp[action_id]
            fn = self.per_action_fn[action_id]

            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)

            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            per_action_metrics[action_id] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
            }

        return per_action_metrics


class DualMetrics:
    """
    Compute both frame-level and interval-level metrics
    Frame-level: Standard classification metrics (fast, for training monitoring)
    Interval-level: Competition metrics (accurate, for final evaluation)
    """

    def __init__(
        self,
        num_actions: int = 4,
        iou_threshold: float = 0.5,
        min_interval_duration: int = 5,
    ):
        self.num_actions = num_actions
        self.iou_threshold = iou_threshold
        self.min_interval_duration = min_interval_duration

        # Frame-level accumulators
        self.frame_correct = 0
        self.frame_total = 0
        self.frame_per_class_correct = {i: 0 for i in range(num_actions)}
        self.frame_per_class_total = {i: 0 for i in range(num_actions)}

        # Interval-level metrics
        self.interval_metrics = IntervalMetrics(
            iou_threshold=iou_threshold,
            match_action=True,
            num_actions=num_actions
        )

    def reset(self):
        """Reset all accumulators"""
        self.frame_correct = 0
        self.frame_total = 0
        self.frame_per_class_correct = {i: 0 for i in range(self.num_actions)}
        self.frame_per_class_total = {i: 0 for i in range(self.num_actions)}
        self.interval_metrics.reset()

    def update_frame_level(self, preds: np.ndarray, targets: np.ndarray):
        """
        Update frame-level metrics

        Args:
            preds: [B, T] or [N] array of predictions
            targets: [B, T] or [N] array of targets
        """
        preds = preds.flatten()
        targets = targets.flatten()

        correct = (preds == targets)
        self.frame_correct += correct.sum()
        self.frame_total += len(preds)

        # Per-class accuracy
        for action_id in range(self.num_actions):
            mask = (targets == action_id)
            if mask.sum() > 0:
                self.frame_per_class_correct[action_id] += correct[mask].sum()
                self.frame_per_class_total[action_id] += mask.sum()

    def update_interval_level(
        self,
        pred_intervals: List[List[Dict]],
        gt_intervals: List[List[Dict]],
    ):
        """
        Update interval-level metrics

        Args:
            pred_intervals: List of predicted intervals per sample
            gt_intervals: List of ground truth intervals per sample
        """
        self.interval_metrics.update(pred_intervals, gt_intervals)

    def compute_frame_metrics(self) -> Dict[str, float]:
        """Compute frame-level metrics"""
        if self.frame_total == 0:
            return {'frame_accuracy': 0.0}

        frame_accuracy = self.frame_correct / self.frame_total

        per_class_acc = {}
        for action_id in range(self.num_actions):
            if self.frame_per_class_total[action_id] > 0:
                acc = self.frame_per_class_correct[action_id] / self.frame_per_class_total[action_id]
                per_class_acc[f'class_{action_id}_acc'] = acc

        return {
            'frame_accuracy': frame_accuracy,
            **per_class_acc
        }

    def compute_interval_metrics(self) -> Dict[str, float]:
        """Compute interval-level metrics"""
        return self.interval_metrics.compute()

    def compute_interval_per_action(self) -> Dict[int, Dict[str, float]]:
        """Compute per-action interval metrics"""
        return self.interval_metrics.compute_per_action()

    def compute_all(self) -> Dict[str, Dict]:
        """
        Compute all metrics

        Returns:
            Dict with 'frame' and 'interval' keys
        """
        return {
            'frame': self.compute_frame_metrics(),
            'interval': self.compute_interval_metrics(),
            'interval_per_action': self.compute_interval_per_action(),
        }
