import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerBehaviorModel(nn.Module):
    """Transformer-based model for mouse behavior recognition"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=4,
                 num_heads=8, num_classes=10, dropout=0.1, max_seq_len=256):
        """
        Args:
            input_dim: Dimension of input features (e.g., number of keypoints * 2)
            hidden_dim: Dimension of hidden layers
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            num_classes: Number of behavior classes
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask tensor

        Returns:
            Output tensor of shape [batch_size, seq_len, num_classes]
        """
        # Project input
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]

        # Transpose for transformer: [seq_len, batch, hidden_dim]
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        # Transpose back: [batch, seq_len, hidden_dim]
        x = x.transpose(0, 1)

        # Classification
        output = self.fc(x)

        return output


class LSTMBehaviorModel(nn.Module):
    """LSTM-based model for mouse behavior recognition"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=2,
                 num_classes=10, dropout=0.1, bidirectional=True):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of LSTM layers
            num_classes: Number of behavior classes
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Output layers
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Output tensor of shape [batch_size, seq_len, num_classes]
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_output_dim]

        # Classification
        output = self.fc(lstm_out)

        return output


def build_model(config):
    """Build model based on configuration"""

    model_type = config.get('model_type', 'transformer')
    input_dim = config['input_dim']
    num_classes = config['num_classes']

    if model_type == 'transformer':
        model = TransformerBehaviorModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            num_heads=config.get('num_heads', 8),
            num_classes=num_classes,
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('sequence_length', 256)
        )
    elif model_type == 'lstm':
        model = LSTMBehaviorModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            num_classes=num_classes,
            dropout=config.get('dropout', 0.1),
            bidirectional=config.get('bidirectional', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model