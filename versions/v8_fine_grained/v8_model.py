"""
V8 Multi-task Behavior Detection Model
Predicts: Action (28 classes) + Agent (4 mice) + Target (4 mice)
"""

import torch
import torch.nn as nn
from .action_mapping import NUM_ACTIONS


class V8BehaviorDetector(nn.Module):
    """
    Multi-task model for fine-grained behavior detection

    Outputs:
        action_logits: [B, T, 28] - behavior classification
        agent_logits: [B, T, 4] - which mouse is the agent
        target_logits: [B, T, 4] - which mouse is the target
    """

    def __init__(
        self,
        input_dim=288,
        num_actions=NUM_ACTIONS,
        num_mice=4,
        conv_channels=[128, 256, 512],
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_mice = num_mice

        # Shared convolutional backbone
        conv_layers = []
        in_channels = input_dim

        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv_backbone = nn.Sequential(*conv_layers)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden * 2  # Bidirectional

        # Task-specific prediction heads

        # Action classification head (most important)
        self.action_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)
        )

        # Agent identification head
        self.agent_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_mice)
        )

        # Target identification head
        self.target_head = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_mice)
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features

        Returns:
            action_logits: [B, T, num_actions]
            agent_logits: [B, T, num_mice]
            target_logits: [B, T, num_mice]
        """
        batch_size, seq_len, _ = x.shape

        # Conv expects [B, D, T]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv_backbone(x)  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]

        # LSTM
        x, _ = self.lstm(x)  # [B, T, lstm_output_dim]

        # Task-specific predictions
        action_logits = self.action_head(x)   # [B, T, num_actions]
        agent_logits = self.agent_head(x)     # [B, T, num_mice]
        target_logits = self.target_head(x)   # [B, T, num_mice]

        return action_logits, agent_logits, target_logits


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, T, C] or [N, C] logits
            targets: [B, T] or [N] class labels
        """
        # Flatten if needed
        if inputs.dim() == 3:
            B, T, C = inputs.shape
            inputs = inputs.view(-1, C)
            targets = targets.view(-1)

        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )

        p = torch.exp(-ce_loss)
        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)  # Prevent numerical issues
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # Check for NaN and replace with mean
        if torch.isnan(focal_loss).any():
            focal_loss = torch.where(torch.isnan(focal_loss), torch.zeros_like(focal_loss), focal_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class V8MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    """

    def __init__(
        self,
        action_weight=1.0,
        agent_weight=0.3,
        target_weight=0.3,
        use_focal=True,
        focal_gamma=2.0,
        class_weights=None
    ):
        super().__init__()

        self.action_weight = action_weight
        self.agent_weight = agent_weight
        self.target_weight = target_weight

        if use_focal:
            self.action_criterion = FocalLoss(
                alpha=class_weights,
                gamma=focal_gamma
            )
        else:
            self.action_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Agent/target use standard CE
        self.agent_criterion = nn.CrossEntropyLoss()
        self.target_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        action_logits,
        agent_logits,
        target_logits,
        action_labels,
        agent_labels,
        target_labels
    ):
        """
        Args:
            action_logits: [B, T, num_actions]
            agent_logits: [B, T, 4]
            target_logits: [B, T, 4]
            action_labels: [B, T]
            agent_labels: [B, T]
            target_labels: [B, T]

        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        # Flatten for loss computation
        B, T = action_labels.shape

        action_logits_flat = action_logits.reshape(-1, action_logits.shape[-1])
        agent_logits_flat = agent_logits.reshape(-1, agent_logits.shape[-1])
        target_logits_flat = target_logits.reshape(-1, target_logits.shape[-1])

        action_labels_flat = action_labels.reshape(-1)
        agent_labels_flat = agent_labels.reshape(-1)
        target_labels_flat = target_labels.reshape(-1)

        # Compute individual losses
        action_loss = self.action_criterion(action_logits_flat, action_labels_flat)
        agent_loss = self.agent_criterion(agent_logits_flat, agent_labels_flat)
        target_loss = self.target_criterion(target_logits_flat, target_labels_flat)

        # Debug: Check for NaN in individual losses
        if torch.isnan(action_loss):
            print(f"[WARNING] action_loss is NaN")
        if torch.isnan(agent_loss):
            print(f"[WARNING] agent_loss is NaN")
        if torch.isnan(target_loss):
            print(f"[WARNING] target_loss is NaN")

        # Combined loss
        total_loss = (
            self.action_weight * action_loss +
            self.agent_weight * agent_loss +
            self.target_weight * target_loss
        )

        if torch.isnan(total_loss):
            print(f"[WARNING] total_loss is NaN! action={action_loss.item():.4f}, agent={agent_loss.item():.4f}, target={target_loss.item():.4f}")

        # Detach for logging (prevent NaN propagation to display)
        loss_dict = {
            'total': total_loss.detach().item() if not torch.isnan(total_loss).any() else 0.0,
            'action': action_loss.detach().item() if not torch.isnan(action_loss).any() else 0.0,
            'agent': agent_loss.detach().item() if not torch.isnan(agent_loss).any() else 0.0,
            'target': target_loss.detach().item() if not torch.isnan(target_loss).any() else 0.0
        }

        return total_loss, loss_dict
