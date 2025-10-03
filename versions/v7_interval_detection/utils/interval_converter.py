"""
Convert frame-level predictions to interval format for competition submission
"""

import numpy as np
import torch
from typing import List, Dict, Tuple


def frame_predictions_to_intervals(
    frame_preds: np.ndarray,
    frame_probs: np.ndarray = None,
    min_duration: int = 5,
    min_confidence: float = 0.5,
) -> List[Dict]:
    """
    Convert frame-level predictions to interval format

    Args:
        frame_preds: [T] array of predicted class IDs (0=background, 1-3=behaviors)
        frame_probs: [T, num_classes] array of prediction probabilities (optional)
        min_duration: Minimum interval duration in frames (filter out short spurious detections)
        min_confidence: Minimum average confidence to keep interval

    Returns:
        intervals: List of dicts with keys:
            - start_frame: int
            - end_frame: int
            - action_id: int (0-3 for V6, or 0-36 for 37 classes)
            - agent_id: int (1-4, currently heuristic)
            - target_id: int (1-4, currently heuristic)
            - confidence: float (average probability in interval)
    """
    intervals = []

    if len(frame_preds) == 0:
        return intervals

    # Track current interval
    current_action = frame_preds[0]
    start_frame = 0

    for frame_idx in range(1, len(frame_preds)):
        # Detect action change
        if frame_preds[frame_idx] != current_action:
            # Save previous interval (exclude background class 0)
            if current_action > 0:
                duration = frame_idx - start_frame

                # Calculate average confidence for this interval
                if frame_probs is not None:
                    interval_probs = frame_probs[start_frame:frame_idx, current_action]
                    avg_confidence = float(np.mean(interval_probs))
                else:
                    avg_confidence = 1.0

                # Filter by duration and confidence
                if duration >= min_duration and avg_confidence >= min_confidence:
                    intervals.append({
                        'start_frame': int(start_frame),
                        'end_frame': int(frame_idx - 1),
                        'action_id': int(current_action),
                        'agent_id': 1,  # TODO: Infer from keypoints
                        'target_id': 2,  # TODO: Infer from keypoints
                        'confidence': avg_confidence,
                    })

            # Start new interval
            current_action = frame_preds[frame_idx]
            start_frame = frame_idx

    # Save last interval
    if current_action > 0:
        duration = len(frame_preds) - start_frame

        if frame_probs is not None:
            interval_probs = frame_probs[start_frame:, current_action]
            avg_confidence = float(np.mean(interval_probs))
        else:
            avg_confidence = 1.0

        if duration >= min_duration and avg_confidence >= min_confidence:
            intervals.append({
                'start_frame': int(start_frame),
                'end_frame': int(len(frame_preds) - 1),
                'action_id': int(current_action),
                'agent_id': 1,
                'target_id': 2,
                'confidence': avg_confidence,
            })

    return intervals


def batch_frame_to_intervals(
    batch_preds: torch.Tensor,
    batch_probs: torch.Tensor = None,
    batch_targets: List[List[Dict]] = None,
    min_duration: int = 5,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    Convert batch of frame predictions to intervals

    Args:
        batch_preds: [B, T] tensor of predictions
        batch_probs: [B, T, C] tensor of probabilities
        batch_targets: List of ground truth intervals per sample (optional)
        min_duration: Minimum interval duration

    Returns:
        pred_intervals: List of predicted intervals per sample
        gt_intervals: List of ground truth intervals per sample (if provided)
    """
    batch_size = batch_preds.shape[0]
    pred_intervals = []

    for b in range(batch_size):
        frame_preds = batch_preds[b].cpu().numpy()

        if batch_probs is not None:
            frame_probs = batch_probs[b].cpu().numpy()
        else:
            frame_probs = None

        intervals = frame_predictions_to_intervals(
            frame_preds,
            frame_probs,
            min_duration=min_duration
        )
        pred_intervals.append(intervals)

    # Convert frame-level targets to intervals if provided
    gt_intervals = batch_targets if batch_targets is not None else None

    return pred_intervals, gt_intervals


def labels_to_intervals(
    frame_labels: np.ndarray,
) -> List[Dict]:
    """
    Convert dense frame labels to interval format (for ground truth)

    Args:
        frame_labels: [T] array of ground truth class IDs

    Returns:
        intervals: List of ground truth intervals
    """
    return frame_predictions_to_intervals(
        frame_labels,
        frame_probs=None,
        min_duration=1,  # No filtering for ground truth
        min_confidence=0.0
    )


def compute_iou(interval1: Dict, interval2: Dict) -> float:
    """
    Compute temporal IoU between two intervals

    Args:
        interval1, interval2: Dicts with 'start_frame' and 'end_frame'

    Returns:
        iou: Intersection over Union (0-1)
    """
    start1, end1 = interval1['start_frame'], interval1['end_frame']
    start2, end2 = interval2['start_frame'], interval2['end_frame']

    # Compute intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    if intersection_start >= intersection_end:
        return 0.0

    intersection = intersection_end - intersection_start + 1

    # Compute union
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection

    return intersection / union if union > 0 else 0.0


def match_intervals(
    pred_intervals: List[Dict],
    gt_intervals: List[Dict],
    iou_threshold: float = 0.5,
    match_action: bool = True,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predicted intervals to ground truth intervals

    Args:
        pred_intervals: List of predicted intervals
        gt_intervals: List of ground truth intervals
        iou_threshold: Minimum IoU to consider a match
        match_action: If True, also require action_id to match

    Returns:
        matches: List of (pred_idx, gt_idx) tuples
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    if len(pred_intervals) == 0 or len(gt_intervals) == 0:
        return [], list(range(len(pred_intervals))), list(range(len(gt_intervals)))

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_intervals), len(gt_intervals)))

    for i, pred in enumerate(pred_intervals):
        for j, gt in enumerate(gt_intervals):
            # Check action match if required
            if match_action and pred['action_id'] != gt['action_id']:
                iou_matrix[i, j] = 0.0
            else:
                iou_matrix[i, j] = compute_iou(pred, gt)

    # Greedy matching: match highest IoU pairs first
    matches = []
    matched_preds = set()
    matched_gts = set()

    # Flatten and sort by IoU
    pairs = []
    for i in range(len(pred_intervals)):
        for j in range(len(gt_intervals)):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((i, j, iou_matrix[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    # Greedily match
    for pred_idx, gt_idx, iou in pairs:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matches.append((pred_idx, gt_idx))
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)

    # Find unmatched
    unmatched_preds = [i for i in range(len(pred_intervals)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_intervals)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts
