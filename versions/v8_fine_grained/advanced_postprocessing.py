"""
V8 Advanced Post-processing for Interval Detection
Implements probability-based inference and temporal smoothing
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.ndimage import median_filter
from collections import defaultdict


# Class-specific configurations (calibrated for higher recall; tune per val)
CLASS_CONFIG = {
    # Sniffing behaviors
    'sniff': {'min_duration': 6, 'prob_threshold': 0.38, 'merge_gap': 5},
    'sniffgenital': {'min_duration': 3, 'prob_threshold': 0.27, 'merge_gap': 5, 'frac_high_threshold': 0.20},
    'sniffface': {'min_duration': 3, 'prob_threshold': 0.28, 'merge_gap': 5, 'frac_high_threshold': 0.20},
    'sniffbody': {'min_duration': 4, 'prob_threshold': 0.30, 'merge_gap': 5, 'frac_high_threshold': 0.20},
    'reciprocalsniff': {'min_duration': 4, 'prob_threshold': 0.40, 'merge_gap': 5},

    # Mating behaviors (favor precision a bit)
    'mount': {'min_duration': 4, 'prob_threshold': 0.40, 'merge_gap': 8},
    'intromit': {'min_duration': 3, 'prob_threshold': 0.40, 'merge_gap': 10},
    'attemptmount': {'min_duration': 3, 'prob_threshold': 0.35, 'merge_gap': 5},
    'ejaculate': {'min_duration': 3, 'prob_threshold': 0.40, 'merge_gap': 3},

    # Aggressive behaviors
    'attack': {'min_duration': 4, 'prob_threshold': 0.40, 'merge_gap': 5},
    'chase': {'min_duration': 5, 'prob_threshold': 0.40, 'merge_gap': 5},
    'chaseattack': {'min_duration': 4, 'prob_threshold': 0.40, 'merge_gap': 4},
    'bite': {'min_duration': 2, 'prob_threshold': 0.35, 'merge_gap': 3},
    'defend': {'min_duration': 4, 'prob_threshold': 0.40, 'merge_gap': 5},

    # Other social behaviors
    'freeze': {'min_duration': 3, 'prob_threshold': 0.25, 'merge_gap': 3},
    'approach': {'min_duration': 4, 'prob_threshold': 0.40, 'merge_gap': 4},
    'follow': {'min_duration': 6, 'prob_threshold': 0.35, 'merge_gap': 5},
    'escape': {'min_duration': 5, 'prob_threshold': 0.55, 'merge_gap': 4, 'max_required': 0.65},
    'shepherd': {'min_duration': 5, 'prob_threshold': 0.35, 'merge_gap': 5},
    'dominancegroom': {'min_duration': 4, 'prob_threshold': 0.35, 'merge_gap': 5},

    # Default for other classes
    'default': {'min_duration': 4, 'prob_threshold': 0.40, 'merge_gap': 4}
}


def sliding_window_inference(
    model,
    keypoints: np.ndarray,
    sequence_length: int = 100,
    stride: int = 25,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding window inference with probability averaging

    Args:
        model: Trained V8 model
        keypoints: [T, D] keypoint features
        sequence_length: Window size
        stride: Sliding stride
        device: Device for inference

    Returns:
        action_probs: [T, num_actions] averaged probabilities
        agent_probs: [T, num_mice] averaged probabilities
        target_probs: [T, num_mice] averaged probabilities
    """
    model.eval()
    T, D = keypoints.shape

    # Determine number of classes from model
    with torch.no_grad():
        dummy_input = torch.zeros(1, sequence_length, D).to(device)
        action_logits, agent_logits, target_logits = model(dummy_input)
        num_actions = action_logits.shape[-1]
        num_mice = agent_logits.shape[-1]

    # Initialize accumulation arrays
    action_probs_sum = np.zeros((T, num_actions), dtype=np.float32)
    agent_probs_sum = np.zeros((T, num_mice), dtype=np.float32)
    target_probs_sum = np.zeros((T, num_mice), dtype=np.float32)
    count = np.zeros(T, dtype=np.int32)

    # Sliding window
    start_indices = list(range(0, max(1, T - sequence_length + 1), stride))

    # Add final window if needed
    if start_indices[-1] + sequence_length < T:
        start_indices.append(T - sequence_length)

    with torch.no_grad():
        for start in start_indices:
            end = min(start + sequence_length, T)

            # Extract window
            window = keypoints[start:end]

            # Pad if necessary
            if len(window) < sequence_length:
                pad_len = sequence_length - len(window)
                window = np.pad(window, ((0, pad_len), (0, 0)), mode='edge')

            # Convert to tensor [1, seq_len, D]
            window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)

            # Forward pass
            action_logits, agent_logits, target_logits = model(window_tensor)

            # Softmax to get probabilities
            action_probs = torch.softmax(action_logits[0], dim=-1).cpu().numpy()  # [seq_len, num_actions]
            agent_probs = torch.softmax(agent_logits[0], dim=-1).cpu().numpy()
            target_probs = torch.softmax(target_logits[0], dim=-1).cpu().numpy()

            # Accumulate (only valid frames)
            valid_len = end - start
            action_probs_sum[start:end] += action_probs[:valid_len]
            agent_probs_sum[start:end] += agent_probs[:valid_len]
            target_probs_sum[start:end] += target_probs[:valid_len]
            count[start:end] += 1

    # Average probabilities
    action_probs_avg = action_probs_sum / count[:, None]
    agent_probs_avg = agent_probs_sum / count[:, None]
    target_probs_avg = target_probs_sum / count[:, None]

    return action_probs_avg, agent_probs_avg, target_probs_avg


def temporal_smoothing(
    probs: np.ndarray,
    kernel_size: int = 5,
    method: str = 'median'
) -> np.ndarray:
    """
    Temporal smoothing of probability sequences

    Args:
        probs: [T, C] probabilities
        kernel_size: Smoothing kernel size (must be odd)
        method: 'median' or 'conv'

    Returns:
        smoothed_probs: [T, C] smoothed probabilities
    """
    T, C = probs.shape
    smoothed = np.zeros_like(probs)

    if method == 'median':
        # Apply median filter to each class
        for c in range(C):
            smoothed[:, c] = median_filter(probs[:, c], size=kernel_size, mode='reflect')
    elif method == 'conv':
        # 1D convolution smoothing
        from scipy.ndimage import convolve1d
        kernel = np.ones(kernel_size) / kernel_size
        for c in range(C):
            smoothed[:, c] = convolve1d(probs[:, c], kernel, mode='reflect')
    else:
        smoothed = probs

    # Re-normalize
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)  # Avoid division by zero
    smoothed = smoothed / row_sums

    return smoothed


def segment_consistency_correction(
    intervals: List[Dict],
    action_names: Dict[int, str]
) -> List[Dict]:
    """
    Enforce segment-level consistency:
    1. Agent != Target within each segment
    2. Majority voting for agent/target within segment

    Args:
        intervals: List of interval dicts
        action_names: Mapping from action_id to action name

    Returns:
        corrected_intervals: List of corrected intervals
    """
    corrected = []

    for interval in intervals:
        action_id = interval['action_id']
        agent_id = interval['agent_id']
        target_id = interval['target_id']

        # Skip background
        if action_id == 0:
            continue

        # Enforce agent != target
        if agent_id == target_id:
            # This shouldn't happen, but if it does, use frame-level majority
            # For now, skip or use a heuristic
            # Skip this interval as it's invalid
            continue

        corrected.append(interval)

    return corrected


def merge_close_segments(
    intervals: List[Dict],
    action_names: Dict[int, str]
) -> List[Dict]:
    """
    Merge segments of same action/agent/target with small gaps

    Args:
        intervals: List of sorted intervals
        action_names: Mapping from action_id to action name

    Returns:
        merged_intervals: List of merged intervals
    """
    if not intervals:
        return []

    # Sort by start (support both key styles)
    def get_start(item):
        return item.get('start') if item.get('start') is not None else item.get('start_frame')

    def get_end(item):
        return item.get('end') if item.get('end') is not None else item.get('stop_frame')

    sorted_intervals = sorted(intervals, key=lambda x: get_start(x))

    merged = [sorted_intervals[0].copy()]

    for current in sorted_intervals[1:]:
        last = merged[-1]

        # Check if same action, agent, target
        same_action = (current['action_id'] == last['action_id'])
        same_agent = (current['agent_id'] == last['agent_id'])
        same_target = (current['target_id'] == last['target_id'])

        if same_action and same_agent and same_target:
            # Get merge gap for this action
            action_name = action_names.get(current['action_id'], 'default')
            config = CLASS_CONFIG.get(action_name, CLASS_CONFIG['default'])
            merge_gap = config['merge_gap']

            # Check gap
            gap = get_start(current) - get_end(last)

            if gap <= merge_gap:
                # Merge: extend end
                last['end'] = get_end(current)
                # Also keep Kaggle-style keys in sync if present
                last['stop_frame'] = last.get('end') if last.get('end') is not None else get_end(last)
                continue

        # No merge, add as new segment
        merged.append(current.copy())

    return merged


def filter_by_velocity(
    intervals: List[Dict],
    keypoints: np.ndarray,
    action_names: Dict[int, str],
    fps: float = 30.0
) -> List[Dict]:
    """
    Filter freeze intervals by velocity threshold

    Args:
        intervals: List of intervals
        keypoints: [T, D] original keypoint features
        action_names: Mapping from action_id to action name
        fps: Video frame rate

    Returns:
        filtered_intervals: Intervals with velocity filtering applied
    """
    filtered = []

    for interval in intervals:
        action_id = interval['action_id']
        action_name = action_names.get(action_id, 'default')

        # Only apply to freeze
        if action_name == 'freeze':
            config = CLASS_CONFIG.get('freeze', CLASS_CONFIG['default'])
            vel_threshold = config.get('velocity_threshold', 0.5)

            # Compute velocity in this segment (support both key styles)
            start = interval.get('start') if interval.get('start') is not None else interval.get('start_frame')
            end = interval.get('end') if interval.get('end') is not None else interval.get('stop_frame')

            # Extract relevant mouse keypoints
            # Assuming keypoints: [7 bodyparts × 4 mice × 2 coords + motion = 112]
            # We need to compute velocity for the agent mouse
            agent_id = interval['agent_id']

            # Extract agent keypoints (7 bodyparts × 2 coords = 14 dims)
            agent_start = agent_id * 14
            agent_end = agent_start + 14
            agent_kps = keypoints[start:end+1, agent_start:agent_end]

            # Compute frame-to-frame displacement
            if len(agent_kps) > 1:
                displacements = np.diff(agent_kps, axis=0)
                velocities = np.linalg.norm(displacements.reshape(len(displacements), -1), axis=1)
                avg_velocity = np.mean(velocities) * fps  # pixels/sec

                # Only keep if velocity is below threshold
                if avg_velocity > vel_threshold:
                    continue  # Skip this interval

        filtered.append(interval)

    return filtered


def calibrate_mating_behaviors(
    intervals: List[Dict],
    action_names: Dict[int, str]
) -> List[Dict]:
    """
    Calibrate intromit and mount:
    - Within same agent/target pair, prioritize intromit over mount
    - Intromit typically happens during mount, so adjust boundaries

    Args:
        intervals: List of intervals
        action_names: Mapping from action_id to action name

    Returns:
        calibrated_intervals: Calibrated intervals
    """
    # Find mount and intromit intervals
    mount_intervals = []
    intromit_intervals = []
    other_intervals = []

    for interval in intervals:
        action_name = action_names.get(interval['action_id'], 'default')
        if action_name == 'mount':
            mount_intervals.append(interval)
        elif action_name == 'intromit':
            intromit_intervals.append(interval)
        else:
            other_intervals.append(interval)

    # For each intromit, check if it overlaps with mount
    calibrated_mount = []
    used_mounts = set()

    for intromit in intromit_intervals:
        for i, mount in enumerate(mount_intervals):
            if i in used_mounts:
                continue

            # Same pair?
            if (mount['agent_id'] == intromit['agent_id'] and
                mount['target_id'] == intromit['target_id']):

                # Check overlap - support both key formats
                mount_start = mount.get('start') if mount.get('start') is not None else mount.get('start_frame')
                mount_end = mount.get('end') if mount.get('end') is not None else mount.get('stop_frame')
                intromit_start = intromit.get('start') if intromit.get('start') is not None else intromit.get('start_frame')
                intromit_end = intromit.get('end') if intromit.get('end') is not None else intromit.get('stop_frame')

                overlap_start = max(mount_start, intromit_start)
                overlap_end = min(mount_end, intromit_end)

                if overlap_start <= overlap_end:
                    # They overlap - keep both but mark mount as used
                    used_mounts.add(i)

    # Keep unused mounts
    for i, mount in enumerate(mount_intervals):
        if i not in used_mounts:
            calibrated_mount.append(mount)

    # Combine all
    return other_intervals + intromit_intervals + calibrated_mount


def refine_segment_boundaries(
    intervals: List[Dict],
    action_probs: np.ndarray,
    agent_probs: np.ndarray,
    target_probs: np.ndarray
) -> List[Dict]:
    """
    Refine segment boundaries by snapping to high-confidence frames

    Args:
        intervals: List of intervals
        action_probs: [T, num_actions] probabilities
        agent_probs: [T, num_mice] probabilities
        target_probs: [T, num_mice] probabilities

    Returns:
        refined_intervals: Intervals with refined boundaries
    """
    refined = []

    for interval in intervals:
        # Support both {'start','end'} and {'start_frame','stop_frame'}
        start = interval.get('start') if interval.get('start') is not None else interval.get('start_frame')
        end = interval.get('end') if interval.get('end') is not None else interval.get('stop_frame')
        action_id = interval['action_id']

        # Search window for boundary refinement (±2 frames)
        search_window = 2

        # Refine start boundary
        search_start = max(0, start - search_window)
        search_end_start = min(len(action_probs), start + search_window + 1)

        if search_end_start > search_start:
            start_confidences = action_probs[search_start:search_end_start, action_id]
            best_start_offset = np.argmax(start_confidences)
            refined_start = search_start + best_start_offset
        else:
            refined_start = start

        # Refine end boundary
        search_start_end = max(0, end - search_window)
        search_end_end = min(len(action_probs), end + search_window + 1)

        if search_end_end > search_start_end:
            end_confidences = action_probs[search_start_end:search_end_end, action_id]
            best_end_offset = np.argmax(end_confidences)
            refined_end = search_start_end + best_end_offset
        else:
            refined_end = end

        # Ensure valid interval
        if refined_end > refined_start:
            interval_copy = interval.copy()
            interval_copy['start'] = refined_start
            interval_copy['end'] = refined_end
            # Remove any stale keys to avoid confusion
            interval_copy.pop('start_frame', None)
            interval_copy.pop('stop_frame', None)
            refined.append(interval_copy)
        else:
            refined.append(interval)

    return refined


def segment_majority_voting(
    intervals: List[Dict],
    agent_probs: np.ndarray,
    target_probs: np.ndarray
) -> List[Dict]:
    """
    Apply majority voting for agent/target within each segment

    Args:
        intervals: List of intervals
        agent_probs: [T, num_mice] probabilities
        target_probs: [T, num_mice] probabilities

    Returns:
        corrected_intervals: Intervals with corrected agent/target
    """
    corrected = []

    for interval in intervals:
        # Support both {'start','end'} and {'start_frame','stop_frame'}
        start = interval.get('start') if interval.get('start') is not None else interval.get('start_frame')
        end = interval.get('end') if interval.get('end') is not None else interval.get('stop_frame')

        # Get segment probabilities
        segment_agent_probs = agent_probs[start:end+1]  # [seg_len, num_mice]
        segment_target_probs = target_probs[start:end+1]

        # Sum probabilities across time to get most likely agent/target
        agent_sum = segment_agent_probs.sum(axis=0)
        target_sum = segment_target_probs.sum(axis=0)

        best_agent = np.argmax(agent_sum)
        best_target = np.argmax(target_sum)

        # If agent == target, use second best for target
        if best_agent == best_target:
            target_sum[best_agent] = -1  # Exclude
            best_target = np.argmax(target_sum)

        interval_copy = interval.copy()
        interval_copy['agent_id'] = int(best_agent)
        interval_copy['target_id'] = int(best_target)
        corrected.append(interval_copy)

    return corrected


def probs_to_intervals_advanced(
    action_probs: np.ndarray,
    agent_probs: np.ndarray,
    target_probs: np.ndarray,
    action_names: Dict[int, str],
    keypoints: np.ndarray = None,
    smoothing_kernel: int = 3
) -> List[Dict]:
    """
    Convert probabilities to intervals with advanced post-processing

    Args:
        action_probs: [T, num_actions] probabilities
        agent_probs: [T, num_mice] probabilities
        target_probs: [T, num_mice] probabilities
        action_names: Mapping from action_id to action name
        keypoints: Optional [T, D] keypoints for velocity filtering

    Returns:
        intervals: List of interval dicts
    """
    T = len(action_probs)

    # 0) Temporal smoothing for action probabilities to reduce jitter
    if smoothing_kernel and smoothing_kernel >= 3 and smoothing_kernel % 2 == 1:
        action_probs = temporal_smoothing(action_probs, kernel_size=smoothing_kernel, method='median')

    # Get frame-wise predictions
    action_preds = np.argmax(action_probs, axis=-1)  # [T]
    agent_preds = np.argmax(agent_probs, axis=-1)
    target_preds = np.argmax(target_probs, axis=-1)

    # 1) Action-only segmentation (do NOT split on agent/target to avoid fragmentation)
    intervals = []
    start_idx = 0
    current_action = action_preds[0]

    for t in range(1, T + 1):
        if t == T or action_preds[t] != current_action:
            end_idx = t - 1

            if current_action != 0:  # skip background
                # Config for this action
                action_name = action_names.get(current_action, 'default')
                config = CLASS_CONFIG.get(action_name, CLASS_CONFIG['default'])

                # Duration filter
                duration = end_idx - start_idx + 1
                if duration >= config['min_duration']:
                    # Prob-based filters (more recall-friendly)
                    seg_probs = action_probs[start_idx:end_idx+1, current_action]
                    avg_prob = float(np.mean(seg_probs))
                    max_prob = float(np.max(seg_probs))
                    frac_high = float(np.mean(seg_probs >= config['prob_threshold']))

                    # Class-adaptive acceptance thresholds
                    frac_high_threshold = float(config.get('frac_high_threshold', 0.30))
                    max_required = float(config.get('max_required', max(0.45, config['prob_threshold'] + 0.10)))

                    # Accept if segment has reasonable evidence
                    if (avg_prob >= config['prob_threshold']) or (max_prob >= max_required) or (frac_high >= frac_high_threshold):
                        intervals.append({
                            'start': start_idx,
                            'end': end_idx,
                            'action_id': int(current_action),
                            # agent/target will be set by majority voting below
                            'agent_id': -1,
                            'target_id': -1,
                            'confidence': avg_prob
                        })

            if t < T:
                start_idx = t
                current_action = action_preds[t]

    # 2) Segment-level majority voting for agent/target (and enforce agent != target inside)
    intervals = segment_majority_voting(intervals, agent_probs, target_probs)

    # 3) Merge close segments (same action/agent/target)
    intervals = merge_close_segments(intervals, action_names)

    # 4) Refine boundaries (snap to high-confidence frames)
    intervals = refine_segment_boundaries(intervals, action_probs, agent_probs, target_probs)

    # 5) Segment consistency (enforce agent != target)
    intervals = segment_consistency_correction(intervals, action_names)

    # 6) Velocity filtering for freeze (if keypoints provided)
    if keypoints is not None:
        intervals = filter_by_velocity(intervals, keypoints, action_names)

    # 7) Calibrate mating behaviors (intromit/mount)
    intervals = calibrate_mating_behaviors(intervals, action_names)

    # 8) Convert to Kaggle-style keys for downstream evaluation
    kaggle_intervals = []
    for interval in intervals:
        item = interval.copy()
        start = item.get('start') if item.get('start') is not None else item.get('start_frame')
        end = item.get('end') if item.get('end') is not None else item.get('stop_frame')
        item['start_frame'] = int(start)
        item['stop_frame'] = int(end)
        # Keep 'start'/'end' too, but ensure consistency
        item['start'] = int(start)
        item['end'] = int(end)
        kaggle_intervals.append(item)

    return kaggle_intervals
