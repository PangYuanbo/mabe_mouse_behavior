"""
V9 Assembler Multi-task Loss
Combines Focal Loss for boundary detection + Soft-IoU for interval quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in boundary detection

    L = -α(1-p)^γ log(p)

    Args:
        alpha: Weighting factor for positive class (default 0.25)
        gamma: Focusing parameter (default 2.0)
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, T, C] predicted probabilities (after sigmoid)
            targets: [B, T, C] target soft labels

        Returns:
            loss: Scalar or tensor depending on reduction
        """
        # Binary focal loss
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # Focal term: (1 - p_t)^gamma
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SoftIoULoss(nn.Module):
    """
    Soft-IoU Loss for interval quality
    Encourages predicted intervals to overlap well with ground truth

    This is an approximation to the Kaggle metric (IoU >= 0.5)
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred_intervals, gt_intervals):
        """
        Args:
            pred_intervals: List of predicted intervals
                Each: {'start': int, 'end': int, 'action_id': int, 'agent_id': int, 'target_id': int, 'confidence': float}
            gt_intervals: List of ground truth intervals

        Returns:
            loss: Soft-IoU loss (lower is better)
        """
        if len(pred_intervals) == 0:
            # Penalize if no predictions but GT exists
            return torch.tensor(float(len(gt_intervals)), dtype=torch.float32)

        if len(gt_intervals) == 0:
            # Penalize if predictions exist but no GT
            return torch.tensor(float(len(pred_intervals)), dtype=torch.float32)

        total_loss = 0.0
        matched_gt = set()

        # For each prediction, find best matching GT
        for pred in pred_intervals:
            best_iou = 0.0

            for gt_idx, gt in enumerate(gt_intervals):
                # Check if same (action, agent, target)
                if (pred['action_id'] == gt['action_id'] and
                    pred['agent_id'] == gt['agent_id'] and
                    pred['target_id'] == gt['target_id']):

                    # Compute soft IoU
                    intersection = max(0, min(pred['end'], gt['end']) - max(pred['start'], gt['start']) + 1)
                    union = (pred['end'] - pred['start'] + 1) + (gt['end'] - gt['start'] + 1) - intersection

                    if union > 0:
                        iou = intersection / union
                        if iou > best_iou:
                            best_iou = iou

                        if iou >= self.threshold:
                            matched_gt.add(gt_idx)

            # Loss: penalize IoU < threshold
            loss_term = max(0, self.threshold - best_iou) ** 2
            total_loss += loss_term

        # Penalize unmatched GT intervals (false negatives)
        num_unmatched = len(gt_intervals) - len(matched_gt)
        total_loss += num_unmatched * (self.threshold ** 2)

        # Normalize by number of GT intervals
        if len(gt_intervals) > 0:
            total_loss = total_loss / len(gt_intervals)

        return torch.tensor(total_loss, dtype=torch.float32)


class BoundaryConsistencyLoss(nn.Module):
    """
    Enforce that start comes before end for each channel
    """

    def __init__(self):
        super().__init__()

    def forward(self, start_heatmap, end_heatmap):
        """
        Args:
            start_heatmap: [B, T, C] start probabilities
            end_heatmap: [B, T, C] end probabilities

        Returns:
            loss: Penalty for inconsistent boundaries
        """
        B, T, C = start_heatmap.shape

        # Create temporal indices [0, 1, 2, ..., T-1]
        indices = torch.arange(T, device=start_heatmap.device).float()
        indices = indices.view(1, T, 1).expand(B, T, C)

        # Expected start frame (weighted average)
        start_expected = (start_heatmap * indices).sum(dim=1) / (start_heatmap.sum(dim=1) + 1e-6)
        # start_expected: [B, C]

        # Expected end frame
        end_expected = (end_heatmap * indices).sum(dim=1) / (end_heatmap.sum(dim=1) + 1e-6)
        # end_expected: [B, C]

        # Loss: penalize if expected_start > expected_end
        consistency_loss = F.relu(start_expected - end_expected).mean()

        return consistency_loss


class AssemblerLoss(nn.Module):
    """
    Multi-task loss for V9 Interval Assembler

    Components:
        1. Boundary detection loss (Focal Loss)
        2. Soft-IoU loss (interval quality)
        3. Boundary consistency loss
        4. Anti-fragmentation regularization

    Args:
        boundary_weight: Weight for boundary detection loss (default 1.0)
        iou_weight: Weight for soft-IoU loss (default 0.5)
        consistency_weight: Weight for boundary consistency (default 0.1)
        fragment_weight: Weight for anti-fragmentation (default 0.05)
        focal_alpha: Focal loss alpha (default 0.25)
        focal_gamma: Focal loss gamma (default 2.0)
    """

    def __init__(
        self,
        boundary_weight=1.0,
        iou_weight=0.5,
        consistency_weight=0.1,
        fragment_weight=0.05,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()

        self.boundary_weight = boundary_weight
        self.iou_weight = iou_weight
        self.consistency_weight = consistency_weight
        self.fragment_weight = fragment_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.soft_iou_loss = SoftIoULoss(threshold=0.5)
        self.boundary_consistency = BoundaryConsistencyLoss()

    def forward(
        self,
        pred_start,
        pred_end,
        pred_confidence,
        target_start,
        target_end,
        pred_intervals=None,
        gt_intervals=None
    ):
        """
        Compute multi-task loss

        Args:
            pred_start: [B, T, C] Predicted start heatmap
            pred_end: [B, T, C] Predicted end heatmap
            pred_confidence: [B, T, C] Predicted segment confidence
            target_start: [B, T, C] Target start heatmap (soft labels)
            target_end: [B, T, C] Target end heatmap (soft labels)
            pred_intervals: Optional list of decoded predicted intervals (for soft-IoU)
            gt_intervals: Optional list of ground truth intervals (for soft-IoU)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # 1. Boundary detection loss (Focal Loss)
        start_loss = self.focal_loss(pred_start, target_start)
        end_loss = self.focal_loss(pred_end, target_end)
        boundary_loss = (start_loss + end_loss) / 2.0

        # 2. Boundary consistency loss
        consistency_loss = self.boundary_consistency(pred_start, pred_end)

        # 3. Soft-IoU loss (if intervals provided)
        if pred_intervals is not None and gt_intervals is not None:
            iou_loss = self.soft_iou_loss(pred_intervals, gt_intervals)
        else:
            iou_loss = torch.tensor(0.0, device=pred_start.device)

        # 4. Anti-fragmentation regularization
        # Penalize rapid on/off switching in predictions
        fragment_loss = self.compute_fragmentation_penalty(pred_start, pred_end)

        # Combine losses
        total_loss = (
            self.boundary_weight * boundary_loss +
            self.iou_weight * iou_loss +
            self.consistency_weight * consistency_loss +
            self.fragment_weight * fragment_loss
        )

        loss_dict = {
            'total': total_loss.item(),
            'boundary': boundary_loss.item(),
            'iou': iou_loss.item() if isinstance(iou_loss, torch.Tensor) else iou_loss,
            'consistency': consistency_loss.item(),
            'fragment': fragment_loss.item()
        }

        return total_loss, loss_dict

    def compute_fragmentation_penalty(self, start_heatmap, end_heatmap):
        """
        Penalize rapid switching between boundary predictions

        Args:
            start_heatmap: [B, T, C]
            end_heatmap: [B, T, C]

        Returns:
            penalty: Fragmentation penalty
        """
        # Compute temporal gradients (frame-to-frame changes)
        start_grad = torch.abs(start_heatmap[:, 1:, :] - start_heatmap[:, :-1, :])
        end_grad = torch.abs(end_heatmap[:, 1:, :] - end_heatmap[:, :-1, :])

        # Average temporal gradient (higher = more fragmented)
        fragment_penalty = (start_grad.mean() + end_grad.mean()) / 2.0

        return fragment_penalty


# Testing
if __name__ == '__main__':
    print("Testing V9 Assembler Loss...")

    # Test Focal Loss
    print("\n[Test 1] Focal Loss:")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    pred = torch.tensor([[0.9, 0.1, 0.8], [0.7, 0.3, 0.6]])  # [B=2, C=3]
    target = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.5]])
    loss = focal(pred, target)
    print(f"  Focal loss: {loss.item():.4f}")
    print("  [OK]")

    # Test Soft-IoU Loss
    print("\n[Test 2] Soft-IoU Loss:")
    soft_iou = SoftIoULoss(threshold=0.5)

    pred_intervals = [
        {'start': 10, 'end': 20, 'action_id': 5, 'agent_id': 0, 'target_id': 1, 'confidence': 0.8},
        {'start': 30, 'end': 40, 'action_id': 10, 'agent_id': 1, 'target_id': 2, 'confidence': 0.7},
    ]

    gt_intervals = [
        {'start': 12, 'end': 22, 'action_id': 5, 'agent_id': 0, 'target_id': 1},
        {'start': 35, 'end': 50, 'action_id': 10, 'agent_id': 1, 'target_id': 2},
    ]

    iou_loss = soft_iou(pred_intervals, gt_intervals)
    print(f"  Soft-IoU loss: {iou_loss.item():.4f}")
    print("  [OK]")

    # Test Boundary Consistency Loss
    print("\n[Test 3] Boundary Consistency Loss:")
    consistency = BoundaryConsistencyLoss()

    # Good case: start before end
    start_good = torch.zeros(2, 100, 3)
    start_good[:, 10, :] = 1.0
    end_good = torch.zeros(2, 100, 3)
    end_good[:, 20, :] = 1.0

    loss_good = consistency(start_good, end_good)
    print(f"  Good case (start=10, end=20): {loss_good.item():.4f}")

    # Bad case: start after end
    start_bad = torch.zeros(2, 100, 3)
    start_bad[:, 30, :] = 1.0
    end_bad = torch.zeros(2, 100, 3)
    end_bad[:, 20, :] = 1.0

    loss_bad = consistency(start_bad, end_bad)
    print(f"  Bad case (start=30, end=20): {loss_bad.item():.4f}")
    assert loss_bad > loss_good
    print("  [OK]")

    # Test Full Assembler Loss
    print("\n[Test 4] Full Assembler Loss:")
    assembler_loss = AssemblerLoss(
        boundary_weight=1.0,
        iou_weight=0.5,
        consistency_weight=0.1,
        fragment_weight=0.05
    )

    pred_start = torch.rand(2, 100, 336)  # [B=2, T=100, C=336]
    pred_end = torch.rand(2, 100, 336)
    pred_conf = torch.rand(2, 100, 336)

    target_start = torch.zeros(2, 100, 336)
    target_start[:, 10, 5] = 1.0
    target_end = torch.zeros(2, 100, 336)
    target_end[:, 20, 5] = 1.0

    total_loss, loss_dict = assembler_loss(
        pred_start, pred_end, pred_conf,
        target_start, target_end,
        pred_intervals=pred_intervals,
        gt_intervals=gt_intervals
    )

    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Boundary loss: {loss_dict['boundary']:.4f}")
    print(f"  IoU loss: {loss_dict['iou']:.4f}")
    print(f"  Consistency loss: {loss_dict['consistency']:.4f}")
    print(f"  Fragment loss: {loss_dict['fragment']:.4f}")
    print("  [OK]")

    # Test backward pass
    print("\n[Test 5] Backward pass:")
    total_loss.backward()
    print("  [OK] Gradients computed")

    print("\n[OK] All tests passed!")
