"""
V7: Temporal Action Detection Model
Outputs behavior intervals directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple


class TemporalActionDetector(nn.Module):
    """
    Temporal Action Detection model for behavior intervals

    Architecture:
    1. Feature extraction (CNN + BiLSTM)
    2. Anchor-based proposal generation
    3. Classification (action + agent/target)
    4. Boundary regression (start/end refinement)
    """

    def __init__(
        self,
        input_dim: int = 142,
        hidden_dim: int = 256,
        num_actions: int = 4,
        num_agents: int = 4,
        sequence_length: int = 1000,
        anchor_scales: List[int] = [10, 30, 60, 120, 240],  # Interval lengths
        iou_threshold: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.sequence_length = sequence_length
        self.anchor_scales = anchor_scales
        self.iou_threshold = iou_threshold

        # Feature extraction
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
        )

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Anchor-based detection heads
        num_anchors = len(anchor_scales)

        # Classification head (action)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_anchors * num_actions)
        )

        # Agent/Target heads
        self.agent_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_anchors * num_agents)
        )

        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_anchors * num_agents)
        )

        # Boundary regression head (offset from anchor)
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_anchors * 2)  # start_offset, end_offset
        )

        # Objectness score (foreground/background)
        self.objectness_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_anchors)
        )

    def generate_anchors(self, sequence_length: int) -> torch.Tensor:
        """
        Generate anchor intervals
        Returns: [num_positions * num_scales, 2] (center, scale)
        """
        anchors = []
        stride = 4  # Downsample factor from conv

        for pos in range(0, sequence_length, stride):
            for scale in self.anchor_scales:
                center = pos
                anchors.append([center, scale])

        return torch.tensor(anchors, dtype=torch.float32)

    def anchors_to_intervals(self, anchors: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Convert anchors + offsets to intervals
        Args:
            anchors: [N, 2] (center, scale)
            offsets: [N, 2] (start_offset, end_offset)
        Returns:
            intervals: [N, 2] (start, end)
        """
        centers = anchors[:, 0]
        scales = anchors[:, 1]

        # Apply offsets
        start = centers - scales / 2 + offsets[:, 0] * scales
        end = centers + scales / 2 + offsets[:, 1] * scales

        # Clamp to valid range
        start = torch.clamp(start, min=0)
        end = torch.clamp(end, max=self.sequence_length - 1)

        return torch.stack([start, end], dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim]

        Returns:
            predictions: dict with keys:
                - 'action_logits': [batch, num_anchors, num_actions]
                - 'agent_logits': [batch, num_anchors, num_agents]
                - 'target_logits': [batch, num_anchors, num_agents]
                - 'boundary_offsets': [batch, num_anchors, 2]
                - 'objectness': [batch, num_anchors]
        """
        batch_size, seq_len, _ = x.shape

        # Feature extraction
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        conv_out = self.conv(x)  # [batch, 256, seq_len//4]
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len//4, 256]

        # Temporal modeling
        lstm_out, _ = self.lstm(conv_out)  # [batch, seq_len//4, hidden*2]

        # Predictions per position
        num_positions = lstm_out.shape[1]
        num_anchors_per_pos = len(self.anchor_scales)

        # Reshape for anchor-based predictions
        # Each position predicts for multiple anchor scales
        lstm_flat = lstm_out.reshape(batch_size * num_positions, -1)

        # Predictions
        action_logits = self.action_head(lstm_flat)  # [B*P, A*C]
        agent_logits = self.agent_head(lstm_flat)
        target_logits = self.target_head(lstm_flat)
        boundary_offsets = self.boundary_head(lstm_flat)
        objectness = self.objectness_head(lstm_flat)

        # Reshape to [batch, num_positions, num_anchors_per_pos, ...]
        action_logits = action_logits.view(batch_size, num_positions, num_anchors_per_pos, self.num_actions)
        agent_logits = agent_logits.view(batch_size, num_positions, num_anchors_per_pos, self.num_agents)
        target_logits = target_logits.view(batch_size, num_positions, num_anchors_per_pos, self.num_agents)
        boundary_offsets = boundary_offsets.view(batch_size, num_positions, num_anchors_per_pos, 2)
        objectness = objectness.view(batch_size, num_positions, num_anchors_per_pos)

        # Flatten anchors dimension
        total_anchors = num_positions * num_anchors_per_pos
        action_logits = action_logits.reshape(batch_size, total_anchors, self.num_actions)
        agent_logits = agent_logits.reshape(batch_size, total_anchors, self.num_agents)
        target_logits = target_logits.reshape(batch_size, total_anchors, self.num_agents)
        boundary_offsets = boundary_offsets.reshape(batch_size, total_anchors, 2)
        objectness = objectness.reshape(batch_size, total_anchors)

        return {
            'action_logits': action_logits,
            'agent_logits': agent_logits,
            'target_logits': target_logits,
            'boundary_offsets': boundary_offsets,
            'objectness': objectness,
        }

    def predict_intervals(
        self,
        predictions: Dict[str, torch.Tensor],
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ) -> List[List[Dict]]:
        """
        Convert model predictions to interval format

        Returns:
            List of predictions per batch item, each containing:
            - start_frame, end_frame, action_id, agent_id, target_id, score
        """
        batch_size = predictions['objectness'].shape[0]
        device = predictions['objectness'].device

        batch_intervals = []

        for b in range(batch_size):
            # Generate anchors for this sequence
            anchors = self.generate_anchors(self.sequence_length).to(device)

            # Get predictions for this item
            objectness = torch.sigmoid(predictions['objectness'][b])
            action_probs = torch.softmax(predictions['action_logits'][b], dim=-1)
            agent_probs = torch.softmax(predictions['agent_logits'][b], dim=-1)
            target_probs = torch.softmax(predictions['target_logits'][b], dim=-1)
            offsets = predictions['boundary_offsets'][b]

            # Filter by objectness score
            keep = objectness > score_threshold

            if keep.sum() == 0:
                batch_intervals.append([])
                continue

            # Get intervals from anchors + offsets
            intervals = self.anchors_to_intervals(anchors[keep], offsets[keep])

            # Get class predictions
            action_ids = torch.argmax(action_probs[keep], dim=-1)
            agent_ids = torch.argmax(agent_probs[keep], dim=-1)
            target_ids = torch.argmax(target_probs[keep], dim=-1)
            scores = objectness[keep]

            # Convert to list of dicts
            item_intervals = []
            for i in range(len(intervals)):
                item_intervals.append({
                    'start_frame': int(intervals[i, 0].item()),
                    'end_frame': int(intervals[i, 1].item()),
                    'action_id': int(action_ids[i].item()),
                    'agent_id': int(agent_ids[i].item()) + 1,  # IDs are 1-indexed
                    'target_id': int(target_ids[i].item()) + 1,
                    'score': float(scores[i].item()),
                })

            # Apply NMS
            item_intervals = self.nms_intervals(item_intervals, nms_threshold)
            batch_intervals.append(item_intervals)

        return batch_intervals

    def nms_intervals(self, intervals: List[Dict], threshold: float) -> List[Dict]:
        """Non-maximum suppression for temporal intervals"""
        if len(intervals) == 0:
            return []

        # Sort by score
        intervals = sorted(intervals, key=lambda x: x['score'], reverse=True)

        keep = []
        while len(intervals) > 0:
            best = intervals.pop(0)
            keep.append(best)

            # Remove overlapping intervals
            new_intervals = []
            for interval in intervals:
                iou = self.compute_iou(
                    [best['start_frame'], best['end_frame']],
                    [interval['start_frame'], interval['end_frame']]
                )
                if iou < threshold:
                    new_intervals.append(interval)

            intervals = new_intervals

        return keep

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
