"""
V8 Kaggle Submission Format Converter
Convert model predictions to Kaggle competition format
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from .action_mapping import ID_TO_ACTION


def predictions_to_intervals(
    action_preds: np.ndarray,
    agent_preds: np.ndarray,
    target_preds: np.ndarray,
    min_duration: int = 5,
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Convert frame-level predictions to interval format

    Args:
        action_preds: [T, num_actions] probabilities or [T] class IDs
        agent_preds: [T, 4] probabilities or [T] agent IDs
        target_preds: [T, 4] probabilities or [T] target IDs
        min_duration: Minimum interval duration in frames
        confidence_threshold: Minimum confidence for action prediction

    Returns:
        List of intervals: [{agent_id, target_id, action, start_frame, stop_frame, confidence}]
    """
    T = len(action_preds)

    # Get class predictions
    if action_preds.ndim == 2:
        action_probs = action_preds
        action_classes = np.argmax(action_preds, axis=1)
        action_confidence = np.max(action_probs, axis=1)
    else:
        action_classes = action_preds
        action_confidence = np.ones(T)

    if agent_preds.ndim == 2:
        agent_classes = np.argmax(agent_preds, axis=1)
    else:
        agent_classes = agent_preds

    if target_preds.ndim == 2:
        target_classes = np.argmax(target_preds, axis=1)
    else:
        target_classes = target_preds

    # Merge consecutive frames into intervals
    intervals = []
    current_action = None
    current_agent = None
    current_target = None
    start_frame = 0
    confidence_sum = 0
    frame_count = 0

    for t in range(T):
        action = action_classes[t]
        agent = agent_classes[t]
        target = target_classes[t]
        conf = action_confidence[t]

        # Skip background (action_id = 0)
        if action == 0:
            # End current interval if exists
            if current_action is not None:
                duration = t - start_frame
                avg_conf = confidence_sum / frame_count if frame_count > 0 else 0

                if duration >= min_duration and avg_conf >= confidence_threshold:
                    intervals.append({
                        'agent_id': current_agent,
                        'target_id': current_target,
                        'action_id': current_action,
                        'action': ID_TO_ACTION.get(current_action, 'background'),
                        'start_frame': start_frame,
                        'stop_frame': t - 1,
                        'confidence': avg_conf
                    })

                # Reset
                current_action = None
                current_agent = None
                current_target = None
                confidence_sum = 0
                frame_count = 0
            continue

        # Check if same interval continues
        if (action == current_action and
            agent == current_agent and
            target == current_target):
            # Continue current interval
            confidence_sum += conf
            frame_count += 1
        else:
            # Save previous interval
            if current_action is not None:
                duration = t - start_frame
                avg_conf = confidence_sum / frame_count if frame_count > 0 else 0

                if duration >= min_duration and avg_conf >= confidence_threshold:
                    intervals.append({
                        'agent_id': current_agent,
                        'target_id': current_target,
                        'action_id': current_action,
                        'action': ID_TO_ACTION.get(current_action, 'background'),
                        'start_frame': start_frame,
                        'stop_frame': t - 1,
                        'confidence': avg_conf
                    })

            # Start new interval
            current_action = action
            current_agent = agent
            current_target = target
            start_frame = t
            confidence_sum = conf
            frame_count = 1

    # Handle last interval
    if current_action is not None:
        duration = T - start_frame
        avg_conf = confidence_sum / frame_count if frame_count > 0 else 0

        if duration >= min_duration and avg_conf >= confidence_threshold:
            intervals.append({
                'agent_id': current_agent,
                'target_id': current_target,
                'action_id': current_action,
                'action': ID_TO_ACTION.get(current_action, 'background'),
                'start_frame': start_frame,
                'stop_frame': T - 1,
                'confidence': avg_conf
            })

    return intervals


def create_submission(
    predictions: Dict[str, Dict],
    video_ids: List[str],
    min_duration: int = 5,
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Create Kaggle submission DataFrame from model predictions

    Args:
        predictions: Dict mapping video_id to prediction dict with:
            - 'action': [T, num_actions] or [T]
            - 'agent': [T, 4] or [T]
            - 'target': [T, 4] or [T]
        video_ids: List of video IDs
        min_duration: Minimum interval duration
        confidence_threshold: Minimum confidence

    Returns:
        DataFrame with columns: [row_id, video_id, agent_id, target_id, action, start_frame, stop_frame]
    """
    all_intervals = []

    for video_id in video_ids:
        if video_id not in predictions:
            continue

        preds = predictions[video_id]
        action_preds = preds['action']
        agent_preds = preds['agent']
        target_preds = preds['target']

        # Convert to numpy if torch tensor
        if torch.is_tensor(action_preds):
            action_preds = action_preds.cpu().numpy()
        if torch.is_tensor(agent_preds):
            agent_preds = agent_preds.cpu().numpy()
        if torch.is_tensor(target_preds):
            target_preds = target_preds.cpu().numpy()

        # Get intervals for this video
        intervals = predictions_to_intervals(
            action_preds=action_preds,
            agent_preds=agent_preds,
            target_preds=target_preds,
            min_duration=min_duration,
            confidence_threshold=confidence_threshold
        )

        # Add video_id to each interval
        for interval in intervals:
            interval['video_id'] = video_id
            # Convert agent/target IDs to mouse names
            interval['agent_id'] = f"mouse{interval['agent_id'] + 1}"
            interval['target_id'] = f"mouse{interval['target_id'] + 1}"

        all_intervals.extend(intervals)

    # Create DataFrame
    if len(all_intervals) == 0:
        # Empty submission
        df = pd.DataFrame(columns=[
            'row_id', 'video_id', 'agent_id', 'target_id',
            'action', 'start_frame', 'stop_frame'
        ])
    else:
        df = pd.DataFrame(all_intervals)
        df['row_id'] = range(len(df))

        # Select and order columns for Kaggle submission
        df = df[[
            'row_id', 'video_id', 'agent_id', 'target_id',
            'action', 'start_frame', 'stop_frame'
        ]]

    return df


def merge_overlapping_intervals(
    intervals: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    Merge overlapping intervals of the same action/agent/target

    Args:
        intervals: List of interval dicts
        iou_threshold: Minimum IoU to consider merging

    Returns:
        Merged intervals
    """
    if len(intervals) == 0:
        return []

    # Sort by start_frame
    intervals = sorted(intervals, key=lambda x: x['start_frame'])

    merged = []
    current = intervals[0].copy()

    for next_interval in intervals[1:]:
        # Check if same action/agent/target
        if (current['action_id'] == next_interval['action_id'] and
            current['agent_id'] == next_interval['agent_id'] and
            current['target_id'] == next_interval['target_id']):

            # Compute IoU
            start1, end1 = current['start_frame'], current['stop_frame']
            start2, end2 = next_interval['start_frame'], next_interval['stop_frame']

            intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
            union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection
            iou = intersection / union if union > 0 else 0

            if iou >= iou_threshold or start2 <= end1 + 5:  # Allow small gap
                # Merge
                current['stop_frame'] = max(end1, end2)
                # Update confidence (weighted average)
                w1 = end1 - start1 + 1
                w2 = end2 - start2 + 1
                current['confidence'] = (
                    current['confidence'] * w1 + next_interval['confidence'] * w2
                ) / (w1 + w2)
                continue

        # Can't merge, save current and start new
        merged.append(current)
        current = next_interval.copy()

    # Add last interval
    merged.append(current)

    return merged


def compute_iou(interval1: Dict, interval2: Dict) -> float:
    """
    Compute temporal IoU between two intervals

    Args:
        interval1, interval2: Dicts with 'start_frame' and 'stop_frame'

    Returns:
        IoU score [0, 1]
    """
    start1, end1 = interval1['start_frame'], interval1['stop_frame']
    start2, end2 = interval2['start_frame'], interval2['stop_frame']

    intersection = max(0, min(end1, end2) - max(start1, start2) + 1)
    union = (end1 - start1 + 1) + (end2 - start2 + 1) - intersection

    return intersection / union if union > 0 else 0


def evaluate_intervals(
    pred_intervals: List[Dict],
    gt_intervals: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute Kaggle-style interval F1 score

    Args:
        pred_intervals: Predicted intervals with keys:
            - 'action_id' or 'action'
            - 'agent_id'
            - 'target_id'
            - 'start_frame'
            - 'stop_frame'
        gt_intervals: Ground truth intervals (same format)
        iou_threshold: Minimum IoU for matching (default 0.5)

    Returns:
        Dict with 'precision', 'recall', 'f1'
    """
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}

    if len(pred_intervals) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_intervals), 'fn': len(gt_intervals)}

    if len(gt_intervals) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_intervals), 'fn': len(gt_intervals)}

    # Match predictions to ground truth
    matched_gt = set()
    true_positives = 0

    for pred in pred_intervals:
        best_iou = 0
        best_gt_idx = -1

        # Extract action ID (handle both 'action_id' and 'action' keys)
        pred_action = pred.get('action_id', pred.get('action'))
        pred_agent = pred.get('agent_id')
        pred_target = pred.get('target_id')

        for gt_idx, gt in enumerate(gt_intervals):
            if gt_idx in matched_gt:
                continue

            # Extract GT action ID
            gt_action = gt.get('action_id', gt.get('action'))
            gt_agent = gt.get('agent_id')
            gt_target = gt.get('target_id')

            # Check if same behavior
            if (pred_action == gt_action and
                pred_agent == gt_agent and
                pred_target == gt_target):

                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        # Check if match is good enough
        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)

    false_positives = len(pred_intervals) - true_positives
    false_negatives = len(gt_intervals) - len(matched_gt)

    # Compute metrics
    precision = true_positives / len(pred_intervals) if len(pred_intervals) > 0 else 0
    recall = true_positives / len(gt_intervals) if len(gt_intervals) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': true_positives,
        'fp': false_positives,
        'fn': false_negatives
    }
