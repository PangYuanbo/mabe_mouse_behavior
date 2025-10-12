"""
V8.6 Aligned Model (No Freeze Branch)
========================================
Aligned with V8.5 training strategy, but with MARS features and architectural improvements:

PRESERVED from V8.6:
- Multi-scale temporal convolutions (3/7/15 kernels)
- Self-attention mechanism
- MARS feature input (370 dims)

REMOVED from V8.6:
- Freeze branch (4th output head)

ALIGNED with V8.5:
- 3 output heads only (action/agent/target)
- Same hyperparameters (lr, batch_size, focal_gamma, etc.)
- No class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv1D(nn.Module):
    """
    Multi-scale 1D convolution to capture different temporal patterns

    Short kernel (3): Fast actions (attack, flinch)
    Medium kernel (7): Normal actions (sniff, mount)
    Long kernel (15): Long behaviors (freeze, intromit)
    """

    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()

        # Three parallel conv branches
        self.conv_short = nn.Sequential(
            nn.Conv1d(input_dim, output_dim // 3, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim // 3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_medium = nn.Sequential(
            nn.Conv1d(input_dim, output_dim // 3, kernel_size=7, padding=3),
            nn.BatchNorm1d(output_dim // 3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_long = nn.Sequential(
            nn.Conv1d(input_dim, output_dim // 3, kernel_size=15, padding=7),
            nn.BatchNorm1d(output_dim // 3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, T]

        Returns:
            out: [B, C_out, T]
        """
        # Parallel convolutions
        short = self.conv_short(x)    # [B, C/3, T]
        medium = self.conv_medium(x)  # [B, C/3, T]
        long = self.conv_long(x)      # [B, C/3, T]

        # Concatenate
        concat = torch.cat([short, medium, long], dim=1)  # [B, C_out, T]

        # Fuse
        out = self.fusion(concat)

        return out


class SelfAttention(nn.Module):
    """
    Self-attention mechanism to focus on critical time windows
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        """
        Args:
            x: [B, T, H]

        Returns:
            out: [B, T, H]
        """
        Q = self.query(x)  # [B, T, H]
        K = self.key(x)    # [B, T, H]
        V = self.value(x)  # [B, T, H]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, T, T]
        attn_weights = F.softmax(scores, dim=-1)  # [B, T, T]

        # Apply attention
        out = torch.matmul(attn_weights, V)  # [B, T, H]

        return out


class V86AlignedBehaviorDetector(nn.Module):
    """
    V8.6 Aligned Multi-task Behavior Detector

    Outputs (SAME AS V8.5):
        action_logits: [B, T, 38] - main behavior classification
        agent_logits: [B, T, 4] - agent mouse identification
        target_logits: [B, T, 4] - target mouse identification
    """

    def __init__(
        self,
        input_dim=370,        # MARS features (V8.5: 112)
        num_actions=38,       # Same as V8.5
        num_mice=4,           # Same as V8.5
        conv_channels=[128, 256, 512],  # SAME AS V8.5 (not 192/384/512!)
        lstm_hidden=256,      # SAME AS V8.5 (not 384!)
        lstm_layers=2,        # SAME AS V8.5 (not 3!)
        dropout=0.0,          # SAME AS V8.5 (not 0.3!)
        use_attention=True    # NEW (but can disable)
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_mice = num_mice
        self.use_attention = use_attention

        # Multi-scale convolutional backbone (NEW in V8.6)
        self.conv1 = MultiScaleConv1D(input_dim, conv_channels[0], dropout)
        self.conv2 = MultiScaleConv1D(conv_channels[0], conv_channels[1], dropout)
        self.conv3 = MultiScaleConv1D(conv_channels[1], conv_channels[2], dropout)

        # Bidirectional LSTM (SAME AS V8.5: 2 layers, hidden=256)
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        lstm_output_dim = lstm_hidden * 2  # Bidirectional

        # Self-attention (NEW in V8.6, optional)
        if use_attention:
            self.attention = SelfAttention(lstm_output_dim)

        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ========================================
        # Task-specific prediction heads (SAME AS V8.5)
        # ========================================

        # 1. Main action classification head
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_actions)  # 38 classes
        )

        # 2. Agent identification head
        self.agent_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_mice)
        )

        # 3. Target identification head
        self.target_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_mice)
        )

        # NO FREEZE HEAD! (Removed to align with V8.5)

    def forward(self, x):
        """
        Args:
            x: [B, T, D] input features (~370 dims)

        Returns (SAME AS V8.5):
            action_logits: [B, T, 38]
            agent_logits: [B, T, 4]
            target_logits: [B, T, 4]
        """
        batch_size, seq_len, _ = x.shape

        # Multi-scale convolutions (expect [B, D, T])
        x = x.transpose(1, 2)  # [B, D, T]

        x = self.conv1(x)  # [B, C1, T]
        x = self.conv2(x)  # [B, C2, T]
        x = self.conv3(x)  # [B, C3, T]

        x = x.transpose(1, 2)  # [B, T, C3]

        # LSTM
        x, _ = self.lstm(x)  # [B, T, lstm_output_dim]

        # Self-attention (optional)
        if self.use_attention:
            x = self.attention(x)  # [B, T, lstm_output_dim]

        # Shared features
        x = self.shared_fc(x)  # [B, T, 512]

        # Task-specific predictions (SAME AS V8.5)
        action_logits = self.action_head(x)   # [B, T, 38]
        agent_logits = self.agent_head(x)     # [B, T, 4]
        target_logits = self.target_head(x)   # [B, T, 4]

        return action_logits, agent_logits, target_logits


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights [num_classes]
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

        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )

        p = torch.exp(-ce_loss)
        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        focal_loss = (1 - p) ** self.gamma * ce_loss

        # Handle NaN
        if torch.isnan(focal_loss).any():
            focal_loss = torch.where(
                torch.isnan(focal_loss),
                torch.zeros_like(focal_loss),
                focal_loss
            )

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class V86AlignedMultiTaskLoss(nn.Module):
    """
    Combined loss for V8.6 Aligned (SAME AS V8.5)

    No freeze-specific loss (removed)
    """

    def __init__(
        self,
        action_weight=1.0,
        agent_weight=0.3,
        target_weight=0.3,
        use_focal=True,
        focal_gamma=1.5,       # SAME AS V8.5 (not 2.0!)
        class_weights=None     # None (no class weights, SAME AS V8.5)
    ):
        super().__init__()

        self.action_weight = action_weight
        self.agent_weight = agent_weight
        self.target_weight = target_weight

        # Action criterion
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
        Args (SAME AS V8.5):
            action_logits: [B, T, 38]
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

        # Combined loss (NO FREEZE LOSS)
        total_loss = (
            self.action_weight * action_loss +
            self.agent_weight * agent_loss +
            self.target_weight * target_loss
        )

        # Detach for logging
        loss_dict = {
            'total': total_loss.detach().item() if not torch.isnan(total_loss).any() else 0.0,
            'action': action_loss.detach().item() if not torch.isnan(action_loss).any() else 0.0,
            'agent': agent_loss.detach().item() if not torch.isnan(agent_loss).any() else 0.0,
            'target': target_loss.detach().item() if not torch.isnan(target_loss).any() else 0.0,
        }

        return total_loss, loss_dict
