"""
Advanced models for MABe mouse behavior detection
Based on research papers and winning solutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBiLSTM(nn.Module):
    """
    1D Convolutional + Bidirectional LSTM Model

    Based on research showing 96% accuracy for mouse behavior classification
    Reference: "Machine Learning in Modeling of Mouse Behavior" (2021)
    """

    def __init__(self, input_dim, num_classes,
                 conv_channels=[64, 128, 256],
                 lstm_hidden=256, lstm_layers=2,
                 dropout=0.3):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of behavior classes
            conv_channels: List of channel sizes for conv layers
            lstm_hidden: Hidden size for LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # 1D Convolutional layers for feature extraction
        # Conv1d operates on [batch, channels, length]
        # Input will be [batch, input_dim, seq_len]
        conv_layers = []
        in_channels = input_dim
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2)
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate output size after pooling
        # After 3 MaxPool1d(kernel_size=2), seq_len is reduced by 2^3 = 8
        self.pooling_factor = 2 ** len(conv_channels)

        # Project to LSTM input size
        self.feature_projection = nn.Linear(conv_channels[-1], lstm_hidden)

        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Output tensor [batch, seq_len, num_classes]
        """
        batch_size, seq_len, input_dim = x.shape

        # Transpose for Conv1d: [batch, input_dim, seq_len]
        x = x.transpose(1, 2)

        # Apply convolutional layers: [batch, channels, seq_len // pooling_factor]
        conv_out = self.conv_layers(x)

        # Transpose back: [batch, seq_len // pooling_factor, channels]
        conv_out = conv_out.transpose(1, 2)

        # Project to LSTM input size: [batch, seq_len // pooling_factor, lstm_hidden]
        lstm_in = self.feature_projection(conv_out)

        # Apply LSTM: [batch, seq_len // pooling_factor, lstm_hidden * 2]
        lstm_out, _ = self.lstm(lstm_in)

        # Upsample to original sequence length using interpolation
        # Reshape for interpolation: [batch, lstm_hidden * 2, seq_len // pooling_factor]
        lstm_out = lstm_out.transpose(1, 2)
        lstm_out = F.interpolate(lstm_out, size=seq_len, mode='linear', align_corners=False)
        # Back to [batch, seq_len, lstm_hidden * 2]
        lstm_out = lstm_out.transpose(1, 2)

        # Classification: [batch, seq_len, num_classes]
        output = self.classifier(lstm_out)

        return output


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN)

    Uses dilated causal convolutions for efficient long-range modeling
    """

    def __init__(self, input_dim, num_classes,
                 num_channels=[64, 128, 256, 256],
                 kernel_size=3, dropout=0.2):
        """
        Args:
            input_dim: Input feature dimension
            num_classes: Number of behavior classes
            num_channels: List of channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]

        Returns:
            Output tensor [batch, seq_len, num_classes]
        """
        # TCN expects [batch, channels, seq_len]
        x = x.transpose(1, 2)

        # Apply TCN
        y = self.network(x)  # [batch, channels, seq_len]

        # Transpose back and classify
        y = y.transpose(1, 2)  # [batch, seq_len, channels]
        output = self.classifier(y)  # [batch, seq_len, num_classes]

        return output


class TemporalBlock(nn.Module):
    """Single temporal block for TCN"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, dilation, padding, dropout=0.2):
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
    """Chomp padding from causal convolution"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class PointNetEncoder(nn.Module):
    """
    PointNet-inspired encoder for mouse keypoints

    Processes keypoints as point clouds with permutation invariance
    """

    def __init__(self, num_keypoints=7, embedding_dim=128):
        """
        Args:
            num_keypoints: Number of keypoints per mouse
            embedding_dim: Output embedding dimension
        """
        super().__init__()

        # Per-point MLP
        self.point_mlp = nn.Sequential(
            nn.Conv1d(2, 64, 1),  # 2D coordinates
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Global feature extraction
        self.global_mlp = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: Keypoints [batch, num_keypoints, 2]

        Returns:
            embedding: [batch, embedding_dim]
        """
        # x: [batch, num_keypoints, 2] -> [batch, 2, num_keypoints]
        x = x.transpose(1, 2)

        # Per-point features
        point_features = self.point_mlp(x)  # [batch, 128, num_keypoints]

        # Global max pooling (permutation invariant)
        global_features = torch.max(point_features, dim=2)[0]  # [batch, 128]

        # Final embedding
        embedding = self.global_mlp(global_features)  # [batch, embedding_dim]

        return embedding


class HybridModel(nn.Module):
    """
    Hybrid model combining PointNet keypoint encoding + Temporal modeling

    Processes each mouse's keypoints through PointNet, then models temporal
    dynamics with LSTM or Transformer
    """

    def __init__(self, num_mice=2, num_keypoints=7, num_classes=4,
                 pointnet_dim=128, temporal_hidden=256, temporal_type='lstm'):
        """
        Args:
            num_mice: Number of mice
            num_keypoints: Keypoints per mouse
            num_classes: Number of behavior classes
            pointnet_dim: PointNet embedding dimension
            temporal_hidden: Hidden size for temporal model
            temporal_type: 'lstm' or 'transformer'
        """
        super().__init__()

        self.num_mice = num_mice
        self.num_keypoints = num_keypoints

        # PointNet encoder for each mouse
        self.pointnet = PointNetEncoder(num_keypoints, pointnet_dim)

        # Temporal modeling
        temporal_input_dim = num_mice * pointnet_dim

        if temporal_type == 'lstm':
            self.temporal_model = nn.LSTM(
                temporal_input_dim,
                temporal_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=0.3
            )
            classifier_input = temporal_hidden * 2
        elif temporal_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=temporal_input_dim,
                nhead=8,
                dim_feedforward=temporal_hidden * 2,
                dropout=0.2,
                batch_first=True
            )
            self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
            classifier_input = temporal_input_dim
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        self.temporal_type = temporal_type

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, temporal_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(temporal_hidden, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Raw keypoints [batch, seq_len, num_mice, num_keypoints, 2]
               or flat [batch, seq_len, num_mice * num_keypoints * 2]

        Returns:
            output: [batch, seq_len, num_classes]
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Reshape if needed
        if len(x.shape) == 3:
            # Flat input: reshape to structured
            x = x.reshape(batch_size, seq_len, self.num_mice, self.num_keypoints, 2)

        # Encode each mouse at each timestep with PointNet
        # x: [batch, seq_len, num_mice, num_keypoints, 2]
        mouse_embeddings = []

        for mouse_idx in range(self.num_mice):
            mouse_kpts = x[:, :, mouse_idx, :, :]  # [batch, seq_len, num_keypoints, 2]

            # Flatten batch and seq for PointNet
            mouse_kpts_flat = mouse_kpts.reshape(-1, self.num_keypoints, 2)

            # Encode
            embeddings_flat = self.pointnet(mouse_kpts_flat)  # [batch*seq_len, pointnet_dim]

            # Reshape back
            embeddings = embeddings_flat.reshape(batch_size, seq_len, -1)
            mouse_embeddings.append(embeddings)

        # Concatenate mouse embeddings
        temporal_input = torch.cat(mouse_embeddings, dim=2)  # [batch, seq_len, num_mice*pointnet_dim]

        # Temporal modeling
        if self.temporal_type == 'lstm':
            temporal_out, _ = self.temporal_model(temporal_input)
        else:  # transformer
            temporal_out = self.temporal_model(temporal_input)

        # Classification
        output = self.classifier(temporal_out)  # [batch, seq_len, num_classes]

        return output


def build_advanced_model(config):
    """Build advanced model based on configuration"""

    model_type = config.get('model_type', 'conv_bilstm')
    input_dim = config['input_dim']
    num_classes = config['num_classes']

    if model_type == 'conv_bilstm':
        model = Conv1DBiLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_channels=config.get('conv_channels', [64, 128, 256]),
            lstm_hidden=config.get('lstm_hidden', 256),
            lstm_layers=config.get('lstm_layers', 2),
            dropout=config.get('dropout', 0.3)
        )
    elif model_type == 'tcn':
        model = TemporalConvNet(
            input_dim=input_dim,
            num_classes=num_classes,
            num_channels=config.get('tcn_channels', [64, 128, 256, 256]),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
    elif model_type == 'hybrid':
        model = HybridModel(
            num_mice=config.get('num_mice', 2),
            num_keypoints=config.get('num_keypoints', 7),
            num_classes=num_classes,
            pointnet_dim=config.get('pointnet_dim', 128),
            temporal_hidden=config.get('temporal_hidden', 256),
            temporal_type=config.get('temporal_model', 'lstm')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model