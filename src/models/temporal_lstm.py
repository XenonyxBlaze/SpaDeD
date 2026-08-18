"""
Temporal Sequence Modeling with Bidirectional LSTM.
Projects multi-attentional frame descriptors and models temporal inter-frame dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        fused_channels: int = 2304,
        bottleneck_dim: int = 512,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        """
        Args:
            num_heads: M attention heads (4)
            fused_channels: C feature channels (2304)
            bottleneck_dim: d_in projection dimension (512)
            lstm_hidden_dim: d_h hidden units per direction (256 -> 512 bidirectional)
            lstm_layers: Number of stacked LSTM layers (2)
            dropout: Dropout rate (0.3)
            num_classes: Number of target classes (2: Real / Fake)
        """
        super().__init__()
        self.num_heads = num_heads
        self.fused_channels = fused_channels
        self.total_input_dim = num_heads * fused_channels  # 4 * 2304 = 9216
        
        # Bottleneck Linear Projection with LayerNorm and GELU
        self.projection = nn.Sequential(
            nn.Linear(self.total_input_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU()
        )
        
        # 2-Layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=bottleneck_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # Sequence-level classification MLP
        lstm_out_dim = lstm_hidden_dim * 2  # 256 * 2 = 512
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, V_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            V_seq: (B, T, M, C) sequence of multi-attentional regional feature vectors
        Returns:
            logits: (B, num_classes) binary classification logits
        """
        B, T, M, C = V_seq.shape
        
        # Flatten M regional vectors per frame: (B, T, M * C) -> (B * T, 9216)
        V_flat = V_seq.view(B * T, M * C)
        
        # Linear Bottleneck Projection: (B * T, 512)
        x_proj = self.projection(V_flat)
        
        # Reshape to temporal sequence: (B, T, 512)
        x_seq = x_proj.view(B, T, -1)
        
        # LSTM Temporal Modeling: (B, T, 512)
        lstm_out, _ = self.lstm(x_seq)
        
        # Temporal Average Pooling over T frames: (B, 512)
        h_seq = torch.mean(lstm_out, dim=1)
        
        # Classification Logits: (B, 2)
        logits = self.classifier(h_seq)
        return logits
