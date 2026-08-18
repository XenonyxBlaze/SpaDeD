"""
Multi-Head Attention and Bilinear Attention Pooling (BAP).
Computes parts-based regional representations and spatial attention distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BilinearAttentionPooling(nn.Module):
    def __init__(self, in_channels: int = 2304, num_heads: int = 4):
        """
        Args:
            in_channels: Dimension C of concatenated feature map F (default: 2048 + 256 = 2304)
            num_heads: Number of attention heads M (default: 4)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        
        # 1x1 conv to compute M attention logits over spatial coordinates
        self.attn_conv = nn.Conv2d(in_channels, num_heads, kernel_size=1, bias=True)

    def forward(self, F_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            F_fused: (B, C, H, W) fused spatial feature tensor
        Returns:
            V_norm: (B, M, C) L2-normalized regional representation vectors
            A: (B, M, H, W) normalized 2D spatial attention maps
        """
        B, C, H, W = F_fused.shape
        
        # Compute spatial attention logits: (B, M, H, W)
        attn_logits = self.attn_conv(F_fused)
        
        # Spatial softmax across (H, W) grid for each head
        attn_logits_flat = attn_logits.view(B, self.num_heads, H * W)
        A_flat = F.softmax(attn_logits_flat, dim=-1)  # (B, M, HW)
        A = A_flat.view(B, self.num_heads, H, W)       # (B, M, H, W)
        
        # Unroll spatial features: (B, C, HW)
        F_flat = F_fused.view(B, C, H * W)
        
        # Bilinear Attention Pooling: V = F * A^T -> (B, C, M)
        # Using batch matrix multiplication: (B, C, HW) x (B, HW, M) -> (B, C, M)
        V = torch.bmm(F_flat, A_flat.transpose(1, 2))  # (B, C, M)
        V = V.transpose(1, 2)                          # (B, M, C)
        
        # L2-normalization across channel dimension C for each regional vector
        V_norm = F.normalize(V, p=2, dim=-1, eps=1e-8)  # (B, M, C)
        
        return V_norm, A
