"""
V9 Interval Assembler Model
Frozen V8 feature extractor + Learnable boundary detection heads
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Import V8 model
sys.path.insert(0, str(Path(__file__).parent.parent))
from v8_fine_grained.v8_model import V8BehaviorDetector


class IntervalAssembler(nn.Module):
    """
    V9 Interval Assembler - Method 1: Learnable Boundary Detector

    Architecture:
        1. Frozen V8 feature extractor (conv + LSTM)
        2. Temporal encoder (BiLSTM/TCN) on V8 outputs
        3. Multi-head outputs:
           - Start boundary heatmap: [T, 28×12]
           - End boundary heatmap: [T, 28×12]
           - Segment confidence: [T, 28×12]

    Args:
        v8_checkpoint: Path to V8 best_model.pth
        v8_config: V8 model configuration dict
        encoder_type: 'bilstm' or 'tcn'
        encoder_hidden: Hidden dimension for temporal encoder
        encoder_layers: Number of encoder layers
        freeze_v8: Whether to freeze V8 weights (default True)
    """

    def __init__(
        self,
        v8_checkpoint: str,
        v8_config: dict,
        encoder_type: str = 'bilstm',
        encoder_hidden: int = 256,
        encoder_layers: int = 2,
        freeze_v8: bool = True,
        num_actions: int = 28,
        num_pairs: int = 12
    ):
        super().__init__()

        self.num_actions = num_actions
        self.num_pairs = num_pairs
        self.num_channels = num_actions * num_pairs  # 28 × 12 = 336
        self.encoder_type = encoder_type
        self.freeze_v8 = freeze_v8

        # Load pre-trained V8 model
        print(f"[V9] Loading V8 checkpoint from {v8_checkpoint}...")
        self.v8_model = V8BehaviorDetector(
            input_dim=v8_config['input_dim'],
            num_actions=num_actions,
            num_mice=v8_config.get('num_mice', 4),
            conv_channels=v8_config['conv_channels'],
            lstm_hidden=v8_config['lstm_hidden'],
            lstm_layers=v8_config['lstm_layers'],
            dropout=v8_config.get('dropout', 0.0)
        )

        # Load V8 weights
        state_dict = torch.load(v8_checkpoint, map_location='cpu')
        self.v8_model.load_state_dict(state_dict)
        print(f"[V9] V8 model loaded successfully")

        # Freeze V8 weights
        if freeze_v8:
            for param in self.v8_model.parameters():
                param.requires_grad = False
            self.v8_model.eval()
            print(f"[V9] V8 weights frozen")

        # V8 outputs 3 logits: action [T, 28], agent [T, 4], target [T, 4]
        # Total V8 feature dimension: 28 + 4 + 4 = 36
        v8_output_dim = num_actions + 4 + 4  # 28 + 4 + 4 = 36

        # Temporal encoder on V8 outputs
        if encoder_type == 'bilstm':
            self.temporal_encoder = nn.LSTM(
                input_size=v8_output_dim,
                hidden_size=encoder_hidden,
                num_layers=encoder_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2 if encoder_layers > 1 else 0.0
            )
            encoder_out_dim = encoder_hidden * 2  # Bidirectional
        elif encoder_type == 'tcn':
            # Temporal Convolutional Network
            self.temporal_encoder = TemporalConvNet(
                num_inputs=v8_output_dim,
                num_channels=[encoder_hidden] * encoder_layers,
                kernel_size=5,
                dropout=0.2
            )
            encoder_out_dim = encoder_hidden
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Multi-head outputs
        # Start boundary detection head
        self.start_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden, self.num_channels),
            nn.Sigmoid()  # [0, 1] probability for each (action, pair) channel
        )

        # End boundary detection head
        self.end_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden, self.num_channels),
            nn.Sigmoid()
        )

        # Segment confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden, self.num_channels),
            nn.Sigmoid()
        )

        print(f"[V9] Assembler model initialized:")
        print(f"  - Encoder: {encoder_type} (hidden={encoder_hidden}, layers={encoder_layers})")
        print(f"  - Output channels: {self.num_channels} (28 actions × 12 pairs)")
        print(f"  - Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, keypoints, return_v8_outputs=False):
        """
        Forward pass

        Args:
            keypoints: [B, T, D] input keypoint features
            return_v8_outputs: If True, also return V8 frame-level predictions

        Returns:
            start_heatmap: [B, T, 336] Start boundary probabilities
            end_heatmap: [B, T, 336] End boundary probabilities
            confidence: [B, T, 336] Segment confidence scores
            (optional) v8_outputs: (action_logits, agent_logits, target_logits)
        """
        B, T, D = keypoints.shape

        # 1. Extract V8 frame-level predictions (frozen)
        with torch.set_grad_enabled(not self.freeze_v8):
            action_logits, agent_logits, target_logits = self.v8_model(keypoints)
            # action_logits: [B, T, 28]
            # agent_logits: [B, T, 4]
            # target_logits: [B, T, 4]

        # 2. Concatenate V8 outputs as features
        # Use logits (not probabilities) to preserve information
        v8_features = torch.cat([action_logits, agent_logits, target_logits], dim=-1)
        # v8_features: [B, T, 36]

        # 3. Temporal encoding
        if self.encoder_type == 'bilstm':
            encoded, _ = self.temporal_encoder(v8_features)
            # encoded: [B, T, encoder_hidden * 2]
        elif self.encoder_type == 'tcn':
            # TCN expects [B, C, T], output [B, C, T]
            encoded = self.temporal_encoder(v8_features.transpose(1, 2)).transpose(1, 2)
            # encoded: [B, T, encoder_hidden]

        # 4. Multi-head outputs
        start_heatmap = self.start_head(encoded)     # [B, T, 336]
        end_heatmap = self.end_head(encoded)         # [B, T, 336]
        confidence = self.confidence_head(encoded)   # [B, T, 336]

        if return_v8_outputs:
            return start_heatmap, end_heatmap, confidence, (action_logits, agent_logits, target_logits)
        else:
            return start_heatmap, end_heatmap, confidence

    def get_v8_predictions(self, keypoints):
        """
        Get V8 frame-level predictions (for evaluation)

        Args:
            keypoints: [B, T, D] input features

        Returns:
            action_preds: [B, T] predicted action IDs
            agent_preds: [B, T] predicted agent IDs
            target_preds: [B, T] predicted target IDs
        """
        with torch.no_grad():
            action_logits, agent_logits, target_logits = self.v8_model(keypoints)

        action_preds = action_logits.argmax(dim=-1)
        agent_preds = agent_logits.argmax(dim=-1)
        target_preds = target_logits.argmax(dim=-1)

        return action_preds, agent_preds, target_preds


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN)
    Alternative to BiLSTM for temporal encoding
    """

    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [B, C_in, T]
        Returns:
            [B, C_out, T]
        """
        return self.network(x)


class TemporalBlock(nn.Module):
    """
    Single temporal block for TCN
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove trailing padding from Conv1d"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


# Testing
if __name__ == '__main__':
    print("Testing V9 Interval Assembler...")

    # Mock V8 config
    v8_config = {
        'input_dim': 112,
        'num_mice': 4,
        'conv_channels': [128, 256, 512],
        'lstm_hidden': 256,
        'lstm_layers': 2,
        'dropout': 0.0
    }

    # Create dummy V8 checkpoint for testing
    print("\n[Test 1] Creating dummy V8 model...")
    dummy_v8 = V8BehaviorDetector(
        input_dim=112,
        num_actions=28,
        num_mice=4,
        conv_channels=[128, 256, 512],
        lstm_hidden=256,
        lstm_layers=2,
        dropout=0.0
    )
    torch.save(dummy_v8.state_dict(), '/tmp/test_v8.pth')
    print("  [OK] Dummy checkpoint saved")

    # Test BiLSTM encoder
    print("\n[Test 2] BiLSTM encoder:")
    model_lstm = IntervalAssembler(
        v8_checkpoint='/tmp/test_v8.pth',
        v8_config=v8_config,
        encoder_type='bilstm',
        encoder_hidden=256,
        encoder_layers=2,
        freeze_v8=True
    )

    dummy_input = torch.randn(2, 100, 112)  # [B=2, T=100, D=112]
    start, end, conf = model_lstm(dummy_input)
    print(f"  Start shape: {start.shape}")
    print(f"  End shape: {end.shape}")
    print(f"  Confidence shape: {conf.shape}")
    assert start.shape == (2, 100, 336)
    assert end.shape == (2, 100, 336)
    assert conf.shape == (2, 100, 336)
    print("  [OK] BiLSTM forward pass")

    # Test TCN encoder
    print("\n[Test 3] TCN encoder:")
    model_tcn = IntervalAssembler(
        v8_checkpoint='/tmp/test_v8.pth',
        v8_config=v8_config,
        encoder_type='tcn',
        encoder_hidden=256,
        encoder_layers=3,
        freeze_v8=True
    )

    start, end, conf = model_tcn(dummy_input)
    print(f"  Start shape: {start.shape}")
    print(f"  End shape: {end.shape}")
    print(f"  Confidence shape: {conf.shape}")
    assert start.shape == (2, 100, 336)
    print("  [OK] TCN forward pass")

    # Test V8 outputs
    print("\n[Test 4] V8 predictions:")
    action_pred, agent_pred, target_pred = model_lstm.get_v8_predictions(dummy_input)
    print(f"  Action preds shape: {action_pred.shape}")
    print(f"  Agent preds shape: {agent_pred.shape}")
    print(f"  Target preds shape: {target_pred.shape}")
    print("  [OK] V8 predictions")

    # Test gradient flow
    print("\n[Test 5] Gradient flow:")
    start, end, conf = model_lstm(dummy_input)
    loss = start.sum() + end.sum() + conf.sum()
    loss.backward()

    v8_has_grad = any(p.grad is not None for p in model_lstm.v8_model.parameters())
    assembler_has_grad = any(p.grad is not None for p in model_lstm.start_head.parameters())

    print(f"  V8 has gradients: {v8_has_grad} (should be False)")
    print(f"  Assembler has gradients: {assembler_has_grad} (should be True)")
    assert not v8_has_grad, "V8 should be frozen!"
    assert assembler_has_grad, "Assembler heads should have gradients!"
    print("  [OK] Gradient flow correct")

    print("\n[OK] All tests passed!")
