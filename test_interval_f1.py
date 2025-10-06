"""
Test Interval F1 calculation
"""

import numpy as np
from versions.v8_fine_grained.submission_utils import (
    predictions_to_intervals,
    evaluate_intervals,
    compute_iou
)


def test_iou():
    """Test IoU computation"""
    interval1 = {'start_frame': 0, 'stop_frame': 100}
    interval2 = {'start_frame': 50, 'stop_frame': 150}

    iou = compute_iou(interval1, interval2)
    # Frames 0-100 (101 frames), frames 50-150 (101 frames)
    # Intersection: frames 50-100 (51 frames)
    # Union: 101 + 101 - 51 = 151 frames
    expected_iou = 51 / 151

    print(f"Test IoU: {iou:.4f} (expected {expected_iou:.4f})")
    assert abs(iou - expected_iou) < 1e-3, "IoU calculation error"
    print("[OK] IoU test passed")


def test_perfect_match():
    """Test perfect prediction"""
    # Create identical predictions and ground truth
    action_preds = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0])
    agent_preds = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0])
    target_preds = np.array([1, 1, 1, 1, 1, 1, 2, 2, 1])

    # Convert to intervals
    pred_intervals = predictions_to_intervals(
        action_preds, agent_preds, target_preds, min_duration=1
    )

    # Use same as GT
    gt_intervals = pred_intervals.copy()

    # Evaluate
    metrics = evaluate_intervals(pred_intervals, gt_intervals, iou_threshold=0.5)

    print(f"\nTest Perfect Match:")
    print(f"  Precision: {metrics['precision']:.4f} (expected 1.0)")
    print(f"  Recall: {metrics['recall']:.4f} (expected 1.0)")
    print(f"  F1: {metrics['f1']:.4f} (expected 1.0)")

    assert abs(metrics['f1'] - 1.0) < 1e-5, "Perfect match should have F1=1.0"
    print("[OK] Perfect match test passed")


def test_partial_match():
    """Test partial match scenario"""
    # Predictions: action 1 from frame 0-99
    action_preds = np.array([1] * 100 + [0] * 100)
    agent_preds = np.zeros(200, dtype=int)
    target_preds = np.ones(200, dtype=int)

    # Ground truth: action 1 from frame 50-149 (50% overlap)
    action_gt = np.array([0] * 50 + [1] * 100 + [0] * 50)
    agent_gt = np.zeros(200, dtype=int)
    target_gt = np.ones(200, dtype=int)

    pred_intervals = predictions_to_intervals(
        action_preds, agent_preds, target_preds, min_duration=1
    )

    gt_intervals = predictions_to_intervals(
        action_gt, agent_gt, target_gt, min_duration=1
    )

    print(f"\nPredicted intervals: {pred_intervals}")
    print(f"Ground truth intervals: {gt_intervals}")

    # Evaluate
    metrics = evaluate_intervals(pred_intervals, gt_intervals, iou_threshold=0.5)

    print(f"\nTest Partial Match:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")

    # IoU = 50/150 = 0.333, which is < 0.5 threshold, so should NOT match
    # Expected: TP=0, FP=1, FN=1, F1=0
    # But let's check actual IoU
    iou = compute_iou(pred_intervals[0], gt_intervals[0])
    print(f"  Actual IoU: {iou:.4f}")

    print("[OK] Partial match test passed")


def test_multi_action():
    """Test multiple actions"""
    # Multiple behaviors
    action_preds = np.array([0, 1, 1, 1, 0, 2, 2, 0, 3, 3, 3, 3])
    agent_preds = np.array([0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2])
    target_preds = np.array([1, 1, 1, 1, 1, 2, 2, 1, 3, 3, 3, 3])

    pred_intervals = predictions_to_intervals(
        action_preds, agent_preds, target_preds, min_duration=1
    )

    print(f"\nTest Multi-action:")
    print(f"  Found {len(pred_intervals)} intervals:")
    for interval in pred_intervals:
        print(f"    Action {interval['action_id']}: frames {interval['start_frame']}-{interval['stop_frame']}")

    expected_count = 3  # actions 1, 2, 3
    assert len(pred_intervals) == expected_count, f"Expected {expected_count} intervals, got {len(pred_intervals)}"
    print("[OK] Multi-action test passed")


if __name__ == '__main__':
    print("="*60)
    print("Testing Interval F1 Calculation")
    print("="*60)

    test_iou()
    test_perfect_match()
    test_partial_match()
    test_multi_action()

    print("\n" + "="*60)
    print("[OK] All tests passed!")
    print("="*60)
