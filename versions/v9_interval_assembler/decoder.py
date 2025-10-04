"""
V9 Decoder - Convert boundary heatmaps to intervals
Peak detection + Greedy pairing + NMS
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from scipy.signal import find_peaks
from .pair_mapping import get_action_pair_from_channel


# Behavior-specific configs (same as V8 advanced postprocessing for consistency)
BEHAVIOR_CONFIGS = {
    'sniff': {'min_duration': 6, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'sniffgenital': {'min_duration': 3, 'conf_threshold': 0.25, 'peak_height': 0.15},
    'sniffface': {'min_duration': 3, 'conf_threshold': 0.25, 'peak_height': 0.15},
    'sniffbody': {'min_duration': 4, 'conf_threshold': 0.25, 'peak_height': 0.15},
    'reciprocalsniff': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'mount': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'intromit': {'min_duration': 3, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'attemptmount': {'min_duration': 3, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'ejaculate': {'min_duration': 3, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'attack': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'chase': {'min_duration': 5, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'chaseattack': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'bite': {'min_duration': 2, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'defend': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'freeze': {'min_duration': 3, 'conf_threshold': 0.25, 'peak_height': 0.15},
    'approach': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'follow': {'min_duration': 6, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'escape': {'min_duration': 5, 'conf_threshold': 0.4, 'peak_height': 0.25},
    'shepherd': {'min_duration': 5, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'dominancegroom': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2},
    'default': {'min_duration': 4, 'conf_threshold': 0.3, 'peak_height': 0.2}
}

# Action ID to name mapping
ID_TO_ACTION = {
    0: 'background', 1: 'attack', 2: 'investigation', 3: 'mount', 4: 'intromit',
    5: 'sniff', 6: 'sniffgenital', 7: 'sniffface', 8: 'sniffbody',
    9: 'approach', 10: 'bite', 11: 'chase', 12: 'chaseattack', 13: 'circle',
    14: 'clean', 15: 'closeinvestigate', 16: 'contact', 17: 'defend',
    18: 'dominancegroom', 19: 'escape', 20: 'flinch', 21: 'follow',
    22: 'freeze', 23: 'grooming', 24: 'other', 25: 'reciprocalsniff',
    26: 'shepherd', 27: 'attemptmount'
}


def detect_peaks(heatmap: np.ndarray, min_height: float = 0.2, min_distance: int = 3) -> List[Tuple[int, float]]:
    """
    Detect peaks in 1D heatmap

    Args:
        heatmap: [T] 1D probability array
        min_height: Minimum peak height
        min_distance: Minimum frames between peaks

    Returns:
        peaks: List of (frame_idx, confidence) tuples
    """
    peaks_idx, properties = find_peaks(heatmap, height=min_height, distance=min_distance)
    peaks = [(int(idx), float(heatmap[idx])) for idx in peaks_idx]
    return peaks


def greedy_pair_boundaries(
    start_peaks: List[Tuple[int, float]],
    end_peaks: List[Tuple[int, float]],
    max_duration: int = 200,
    min_duration: int = 2
) -> List[Tuple[int, int, float]]:
    """
    Greedily pair start and end peaks

    Args:
        start_peaks: List of (frame, conf) for starts
        end_peaks: List of (frame, conf) for ends
        max_duration: Maximum interval duration
        min_duration: Minimum interval duration

    Returns:
        intervals: List of (start_frame, end_frame, avg_conf)
    """
    intervals = []
    used_ends = set()

    for start_frame, start_conf in start_peaks:
        best_end = None
        best_score = -1

        for end_idx, (end_frame, end_conf) in enumerate(end_peaks):
            if end_idx in used_ends:
                continue

            # End must be after start
            if end_frame <= start_frame:
                continue

            # Check duration constraints
            duration = end_frame - start_frame + 1
            if duration < min_duration or duration > max_duration:
                continue

            # Score: average confidence, prefer closer ends
            avg_conf = (start_conf + end_conf) / 2
            distance_penalty = (end_frame - start_frame) / max_duration
            score = avg_conf * (1 - 0.1 * distance_penalty)

            if score > best_score:
                best_score = score
                best_end = (end_idx, end_frame, avg_conf)

        if best_end is not None:
            end_idx, end_frame, avg_conf = best_end
            intervals.append((start_frame, end_frame, avg_conf))
            used_ends.add(end_idx)

    return intervals


def decode_intervals(
    start_heatmap: np.ndarray,
    end_heatmap: np.ndarray,
    confidence_heatmap: np.ndarray,
    num_actions: int = 28
) -> List[Dict]:
    """
    Decode boundary heatmaps into intervals

    Args:
        start_heatmap: [T, C] Start boundary probabilities (C = 28×12 = 336)
        end_heatmap: [T, C] End boundary probabilities
        confidence_heatmap: [T, C] Segment confidence scores
        num_actions: Number of action classes

    Returns:
        intervals: List of interval dicts
    """
    T, C = start_heatmap.shape
    assert C == num_actions * 12

    intervals = []

    # Process each channel (action × pair combination)
    for channel in range(C):
        action_id, agent_id, target_id = get_action_pair_from_channel(channel, num_actions)

        # Skip background
        if action_id == 0:
            continue

        # Get behavior config
        action_name = ID_TO_ACTION.get(action_id, 'default')
        config = BEHAVIOR_CONFIGS.get(action_name, BEHAVIOR_CONFIGS['default'])

        # Detect peaks
        start_peaks = detect_peaks(
            start_heatmap[:, channel],
            min_height=config['peak_height'],
            min_distance=3
        )

        end_peaks = detect_peaks(
            end_heatmap[:, channel],
            min_height=config['peak_height'],
            min_distance=3
        )

        # Pair start/end boundaries
        paired = greedy_pair_boundaries(
            start_peaks, end_peaks,
            max_duration=200,
            min_duration=config['min_duration']
        )

        # Create intervals
        for start_frame, end_frame, boundary_conf in paired:
            # Get average segment confidence
            seg_conf = confidence_heatmap[start_frame:end_frame+1, channel].mean()
            overall_conf = (boundary_conf + seg_conf) / 2

            # Apply confidence threshold
            if overall_conf >= config['conf_threshold']:
                intervals.append({
                    'start_frame': start_frame,
                    'stop_frame': end_frame,
                    'action_id': action_id,
                    'agent_id': agent_id,
                    'target_id': target_id,
                    'confidence': overall_conf
                })

    return intervals


def temporal_nms(intervals: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """
    Non-Maximum Suppression for overlapping intervals

    Args:
        intervals: List of interval dicts with 'confidence'
        iou_threshold: IoU threshold for suppression

    Returns:
        kept_intervals: Filtered intervals
    """
    if not intervals:
        return []

    # Sort by confidence (descending)
    sorted_intervals = sorted(intervals, key=lambda x: x['confidence'], reverse=True)

    kept = []

    while sorted_intervals:
        best = sorted_intervals.pop(0)
        kept.append(best)

        # Remove overlapping intervals with same (action, agent, target)
        remaining = []
        for interval in sorted_intervals:
            # Different behavior or pair -> keep
            if (interval['action_id'] != best['action_id'] or
                interval['agent_id'] != best['agent_id'] or
                interval['target_id'] != best['target_id']):
                remaining.append(interval)
                continue

            # Compute IoU
            intersection = max(0, min(best['stop_frame'], interval['stop_frame']) -
                             max(best['start_frame'], interval['start_frame']) + 1)
            union = ((best['stop_frame'] - best['start_frame'] + 1) +
                    (interval['stop_frame'] - interval['start_frame'] + 1) - intersection)

            iou = intersection / union if union > 0 else 0

            # Keep if IoU is low enough
            if iou < iou_threshold:
                remaining.append(interval)

        sorted_intervals = remaining

    return kept


# Testing
if __name__ == '__main__':
    print("Testing V9 Decoder...")

    # Test peak detection
    print("\n[Test 1] Peak detection:")
    heatmap = np.zeros(100)
    heatmap[10] = 0.8
    heatmap[30] = 0.6
    heatmap[50] = 0.9

    peaks = detect_peaks(heatmap, min_height=0.5)
    print(f"  Detected {len(peaks)} peaks: {peaks}")
    assert len(peaks) == 3
    print("  [OK]")

    # Test greedy pairing
    print("\n[Test 2] Greedy pairing:")
    start_peaks = [(10, 0.8), (30, 0.7)]
    end_peaks = [(20, 0.7), (40, 0.8), (60, 0.6)]

    paired = greedy_pair_boundaries(start_peaks, end_peaks, max_duration=50, min_duration=5)
    print(f"  Paired {len(paired)} intervals: {paired}")
    assert len(paired) == 2
    print("  [OK]")

    # Test full decoding
    print("\n[Test 3] Full decoding:")
    T, C = 100, 336
    start_hm = np.zeros((T, C))
    end_hm = np.zeros((T, C))
    conf_hm = np.zeros((T, C))

    # Add synthetic interval for action=5, pair=0
    channel = 5 * 12 + 0
    start_hm[10, channel] = 0.9
    end_hm[20, channel] = 0.8
    conf_hm[10:21, channel] = 0.7

    intervals = decode_intervals(start_hm, end_hm, conf_hm)
    print(f"  Decoded {len(intervals)} intervals")
    if intervals:
        print(f"  First interval: {intervals[0]}")
    print("  [OK]")

    # Test NMS
    print("\n[Test 4] Temporal NMS:")
    test_intervals = [
        {'start_frame': 10, 'stop_frame': 20, 'action_id': 5, 'agent_id': 0, 'target_id': 1, 'confidence': 0.9},
        {'start_frame': 15, 'stop_frame': 25, 'action_id': 5, 'agent_id': 0, 'target_id': 1, 'confidence': 0.7},
        {'start_frame': 30, 'stop_frame': 40, 'action_id': 5, 'agent_id': 0, 'target_id': 1, 'confidence': 0.8},
    ]

    kept = temporal_nms(test_intervals, iou_threshold=0.3)
    print(f"  Before NMS: {len(test_intervals)} intervals")
    print(f"  After NMS: {len(kept)} intervals")
    assert len(kept) == 2  # Should keep highest conf and non-overlapping
    print("  [OK]")

    print("\n[OK] All tests passed!")
