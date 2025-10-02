"""
V7: Loss functions for interval detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np


class IntervalDetectionLoss(nn.Module):
    """
    Combined loss for temporal action detection:
    - IoU loss for boundary regression
    - Cross-entropy for action classification
    - Cross-entropy for agent/target classification
    - Binary cross-entropy for objectness
    """

    def __init__(
        self,
        iou_threshold_pos: float = 0.5,
        iou_threshold_neg: float = 0.4,
        alpha: float = 0.25,  # Focal loss alpha
        gamma: float = 2.0,   # Focal loss gamma
    ):
        super().__init__()
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[List[Dict]],
        anchors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model outputs
                - action_logits: [batch, num_anchors, num_actions]
                - agent_logits: [batch, num_anchors, num_agents]
                - target_logits: [batch, num_anchors, num_agents]
                - boundary_offsets: [batch, num_anchors, 2]
                - objectness: [batch, num_anchors]
            targets: List of ground truth intervals per batch
            anchors: [num_anchors, 2] (center, scale)

        Returns:
            losses: dict of individual losses
        """
        batch_size = predictions['objectness'].shape[0]
        device = predictions['objectness'].device

        # Assign ground truth to anchors
        matched_targets = self.match_anchors_to_targets(anchors, targets, device)

        # Compute losses
        objectness_loss = self.compute_objectness_loss(
            predictions['objectness'],
            matched_targets['objectness']
        )

        action_loss = self.compute_classification_loss(
            predictions['action_logits'],
            matched_targets['action_ids'],
            matched_targets['objectness']
        )

        agent_loss = self.compute_classification_loss(
            predictions['agent_logits'],
            matched_targets['agent_ids'],
            matched_targets['objectness']
        )

        target_loss = self.compute_classification_loss(
            predictions['target_logits'],
            matched_targets['target_ids'],
            matched_targets['objectness']
        )

        boundary_loss = self.compute_boundary_loss(
            predictions['boundary_offsets'],
            matched_targets['boundary_targets'],
            matched_targets['objectness'],
            anchors
        )

        total_loss = (
            objectness_loss +
            action_loss +
            agent_loss +
            target_loss +
            boundary_loss
        )

        return {
            'total_loss': total_loss,
            'objectness_loss': objectness_loss,
            'action_loss': action_loss,
            'agent_loss': agent_loss,
            'target_loss': target_loss,
            'boundary_loss': boundary_loss,
        }

    def match_anchors_to_targets(
        self,
        anchors: torch.Tensor,
        targets: List[List[Dict]],
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Assign ground truth intervals to anchors based on IoU

        Returns:
            Dict with matched targets for each anchor
        """
        batch_size = len(targets)
        num_anchors = anchors.shape[0]

        objectness = torch.zeros(batch_size, num_anchors, dtype=torch.float32, device=device)
        action_ids = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        agent_ids = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        target_ids = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=device)
        boundary_targets = torch.zeros(batch_size, num_anchors, 2, dtype=torch.float32, device=device)

        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue

            # Convert anchors to intervals
            anchor_intervals = self.anchors_to_intervals(anchors)

            # For each anchor, find best matching GT
            for anchor_idx in range(num_anchors):
                anchor_interval = anchor_intervals[anchor_idx]

                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(targets[b]):
                    gt_interval = [gt['start_frame'], gt['end_frame']]
                    iou = self.compute_iou(anchor_interval, gt_interval)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # Assign if IoU is above threshold
                if best_iou >= self.iou_threshold_pos:
                    objectness[b, anchor_idx] = 1.0
                    gt = targets[b][best_gt_idx]
                    action_ids[b, anchor_idx] = gt['action_id']
                    agent_ids[b, anchor_idx] = gt['agent_id'] - 1  # Convert to 0-indexed
                    target_ids[b, anchor_idx] = gt['target_id'] - 1

                    # Compute boundary targets (offsets from anchor)
                    anchor_center = anchors[anchor_idx, 0]
                    anchor_scale = anchors[anchor_idx, 1]

                    start_offset = (gt['start_frame'] - (anchor_center - anchor_scale / 2)) / anchor_scale
                    end_offset = (gt['end_frame'] - (anchor_center + anchor_scale / 2)) / anchor_scale

                    boundary_targets[b, anchor_idx, 0] = start_offset
                    boundary_targets[b, anchor_idx, 1] = end_offset

                elif best_iou < self.iou_threshold_neg:
                    objectness[b, anchor_idx] = 0.0

                # Else: ignore (neither pos nor neg)

        return {
            'objectness': objectness,
            'action_ids': action_ids,
            'agent_ids': agent_ids,
            'target_ids': target_ids,
            'boundary_targets': boundary_targets,
        }

    def compute_objectness_loss(
        self,
        pred_objectness: torch.Tensor,
        gt_objectness: torch.Tensor
    ) -> torch.Tensor:
        """Focal loss for objectness (foreground/background)"""
        pred_prob = torch.sigmoid(pred_objectness)

        # Focal loss
        pt = gt_objectness * pred_prob + (1 - gt_objectness) * (1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = gt_objectness * self.alpha + (1 - gt_objectness) * (1 - self.alpha)

        loss = -alpha_weight * focal_weight * (
            gt_objectness * torch.log(pred_prob + 1e-8) +
            (1 - gt_objectness) * torch.log(1 - pred_prob + 1e-8)
        )

        return loss.mean()

    def compute_classification_loss(
        self,
        pred_logits: torch.Tensor,
        gt_ids: torch.Tensor,
        objectness_mask: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss for classification (only on positive anchors)"""
        # Only compute loss on positive anchors
        pos_mask = objectness_mask > 0.5

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_logits.device)

        pos_logits = pred_logits[pos_mask]
        pos_targets = gt_ids[pos_mask]

        loss = F.cross_entropy(pos_logits, pos_targets)
        return loss

    def compute_boundary_loss(
        self,
        pred_offsets: torch.Tensor,
        gt_offsets: torch.Tensor,
        objectness_mask: torch.Tensor,
        anchors: torch.Tensor
    ) -> torch.Tensor:
        """IoU loss for boundary regression"""
        pos_mask = objectness_mask > 0.5

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_offsets.device)

        batch_size = pred_offsets.shape[0]
        device = pred_offsets.device

        # Expand anchors for batch
        anchors_expanded = anchors.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        # Get positive predictions and targets
        pos_pred_offsets = pred_offsets[pos_mask]
        pos_gt_offsets = gt_offsets[pos_mask]
        pos_anchors = anchors_expanded[pos_mask]

        # Convert to intervals
        pred_intervals = self.offsets_to_intervals(pos_anchors, pos_pred_offsets)
        gt_intervals = self.offsets_to_intervals(pos_anchors, pos_gt_offsets)

        # Compute IoU loss
        iou_loss = self.iou_loss(pred_intervals, gt_intervals)

        return iou_loss

    @staticmethod
    def anchors_to_intervals(anchors: torch.Tensor) -> List[List[float]]:
        """Convert anchors to interval format"""
        intervals = []
        for anchor in anchors:
            center = anchor[0].item()
            scale = anchor[1].item()
            start = center - scale / 2
            end = center + scale / 2
            intervals.append([start, end])
        return intervals

    @staticmethod
    def offsets_to_intervals(anchors: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Convert anchors + offsets to intervals"""
        centers = anchors[:, 0]
        scales = anchors[:, 1]

        start = centers - scales / 2 + offsets[:, 0] * scales
        end = centers + scales / 2 + offsets[:, 1] * scales

        return torch.stack([start, end], dim=1)

    @staticmethod
    def compute_iou(interval1: List[float], interval2: List[float]) -> float:
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

    @staticmethod
    def iou_loss(pred_intervals: torch.Tensor, gt_intervals: torch.Tensor) -> torch.Tensor:
        """IoU loss for interval regression"""
        # Compute IoU
        pred_start, pred_end = pred_intervals[:, 0], pred_intervals[:, 1]
        gt_start, gt_end = gt_intervals[:, 0], gt_intervals[:, 1]

        intersection_start = torch.max(pred_start, gt_start)
        intersection_end = torch.min(pred_end, gt_end)

        intersection = torch.clamp(intersection_end - intersection_start, min=0)
        union = (pred_end - pred_start) + (gt_end - gt_start) - intersection

        iou = intersection / (union + 1e-8)

        # IoU loss = 1 - IoU
        loss = (1 - iou).mean()

        return loss
