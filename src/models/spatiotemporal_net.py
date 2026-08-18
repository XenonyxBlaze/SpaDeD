"""
Complete Unified Spatiotemporal Deepfake Detection Network.
Integrates ResNeXt50, Texture Enhancement Block (TEB), Multi-Attentional BAP, and BiLSTM.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Optional

from .texture_block import TextureEnhancementBlock
from .multi_attention import BilinearAttentionPooling
from .temporal_lstm import TemporalLSTMClassifier


class SpatiotemporalDeepfakeDetector(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnext50_32x4d",
        pretrained: bool = True,
        num_heads: int = 4,
        texture_channels: int = 256,
        semantic_channels: int = 2048,
        bottleneck_dim: int = 512,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        """
        End-to-End Spatiotemporal Deepfake Detector.
        """
        super().__init__()
        self.num_heads = num_heads
        self.texture_channels = texture_channels
        self.semantic_channels = semantic_channels
        self.fused_channels = semantic_channels + texture_channels  # 2048 + 256 = 2304
        
        # 1. Spatial Backbone (ResNeXt-50 32x4d)
        if backbone_name == "resnext50_32x4d":
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
            base_model = models.resnext50_32x4d(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            
        # Decompose backbone to access shallow and deep feature maps
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1  # Output: (B, 256, 64, 64) -> Shallow Low Features
        self.layer2 = base_model.layer2  # Output: (B, 512, 32, 32)
        self.layer3 = base_model.layer3  # Output: (B, 1024, 16, 16)
        self.layer4 = base_model.layer4  # Output: (B, 2048, 8, 8) -> Deep Semantic Features
        
        # 2. Shallow Texture Enhancement Block (TEB)
        self.texture_block = TextureEnhancementBlock(
            in_channels=256,
            out_channels=texture_channels,
            target_hw=8
        )
        
        # 3. Multi-Head Attention & Bilinear Attention Pooling (BAP)
        self.bap = BilinearAttentionPooling(
            in_channels=self.fused_channels,
            num_heads=num_heads
        )
        
        # 4. Temporal Sequence BiLSTM & Classifier
        self.temporal_classifier = TemporalLSTMClassifier(
            num_heads=num_heads,
            fused_channels=self.fused_channels,
            bottleneck_dim=bottleneck_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            dropout=dropout,
            num_classes=num_classes
        )

    def extract_spatial_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts spatial features for a batch of frames.
        Args:
            x: (B*T, 3, 256, 256)
        Returns:
            V: (B*T, M, C) normalized regional feature vectors
            A: (B*T, M, H, W) spatial attention maps
        """
        # Shallow stem
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        f_low = self.layer1(out)       # (B*T, 256, 64, 64)
        
        # Deep semantic path
        out = self.layer2(f_low)
        out = self.layer3(out)
        f_sem = self.layer4(out)       # (B*T, 2048, 8, 8)
        
        # Shallow texture path
        f_tex = self.texture_block(f_low)  # (B*T, 256, 8, 8)
        
        # Channel-wise concatenation
        f_fused = torch.cat([f_sem, f_tex], dim=1)  # (B*T, 2304, 8, 8)
        
        # Bilinear Attention Pooling
        V, A = self.bap(f_fused)  # V: (B*T, M, 2304), A: (B*T, M, 8, 8)
        return V, A

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: Video sequence (B, T, 3, 256, 256) or static image (B, 3, 256, 256)
            return_attention: Whether to return spatial attention maps and regional features
        Returns:
            logits: (B, 2) classification logits
            V_seq: Optional (B, T, M, C) regional vectors
            A_seq: Optional (B, T, M, H, W) attention maps
        """
        # Handle static image input via temporal replication (T=20)
        if x.dim() == 4:
            # (B, 3, H, W) -> replicate to (B, 20, 3, H, W)
            B, C, H, W = x.shape
            T = 20
            x = x.unsqueeze(1).repeat(1, T, 1, 1, 1)
        else:
            B, T, C, H, W = x.shape
            
        # Fold batch and sequence: (B*T, 3, 256, 256)
        x_flat = x.view(B * T, C, H, W)
        
        # Spatial Phase 2
        V_flat, A_flat = self.extract_spatial_features(x_flat)
        
        # Unfold back to temporal sequence: (B, T, M, C)
        V_seq = V_flat.view(B, T, self.num_heads, self.fused_channels)
        A_seq = A_flat.view(B, T, self.num_heads, A_flat.shape[2], A_flat.shape[3])
        
        # Temporal Phase 3
        logits = self.temporal_classifier(V_seq)
        
        if return_attention:
            return logits, V_seq, A_seq
        return logits, None, None
