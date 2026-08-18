"""
Loss Functions for Spatiotemporal Deepfake Detection.
Implements Cross-Entropy and the Dual-Regularized Regional Independence Loss (L_RIL).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RegionalIndependenceLoss(nn.Module):
    def __init__(self, margin: float = 0.2, gamma_feat: float = 1.0):
        """
        Dual-Regularized Regional Independence Loss (L_RIL).
        Args:
            margin: Margin m for cosine similarity feature penalty
            gamma_feat: Weight for feature orthogonality term
        """
        super().__init__()
        self.margin = margin
        self.gamma_feat = gamma_feat

    def forward(
        self,
        V_seq: torch.Tensor,
        A_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            V_seq: (B, T, M, C) normalized regional feature vectors
            A_seq: (B, T, M, H, W) normalized 2D spatial attention maps
        Returns:
            l_ril: Total Regional Independence Loss
            l_spatial: Spatial attention map non-overlap loss
            l_feat: Regional feature orthogonality loss
        """
        B, T, M, C = V_seq.shape
        _, _, _, H, W = A_seq.shape
        
        # Flatten temporal and batch dimensions: (B*T, M, C) and (B*T, M, H, W)
        V = V_seq.view(B * T, M, C)
        A = A_seq.view(B * T, M, H, W)
        N = B * T
        
        # 1. Spatial Attention Map Non-Overlap Loss (L_spatial)
        # Directly penalizes spatial overlap between attention heads across the (H, W) grid
        l_spatial = torch.tensor(0.0, device=V.device)
        pair_count = 0
        for i in range(M):
            for j in range(i + 1, M):
                # Elementwise multiplication of attention maps integrated over spatial grid
                overlap = torch.sum(A[:, i, :, :] * A[:, j, :, :], dim=(1, 2))  # (N,)
                l_spatial = l_spatial + torch.mean(overlap)
                pair_count += 1
        if pair_count > 0:
            l_spatial = l_spatial / pair_count

        # 2. Regional Feature Orthogonality Loss (L_feat)
        # Cosine similarity between normalized vectors Vi and Vj with margin m
        # (N, M, C) x (N, C, M) -> (N, M, M) cosine similarity matrix
        sim_matrix = torch.bmm(V, V.transpose(1, 2))
        
        l_feat = torch.tensor(0.0, device=V.device)
        for i in range(M):
            for j in range(i + 1, M):
                sim_ij = sim_matrix[:, i, j]
                penalty = F.relu(sim_ij - self.margin)
                l_feat = l_feat + torch.mean(penalty)
        if pair_count > 0:
            l_feat = l_feat / pair_count
            
        # Composite Regional Independence Loss
        l_ril = l_spatial + self.gamma_feat * l_feat
        return l_ril, l_spatial, l_feat


class TotalDetectionLoss(nn.Module):
    def __init__(
        self,
        lambda_ril: float = 0.5,
        margin: float = 0.2,
        gamma_feat: float = 1.0
    ):
        """
        Total End-to-End Objective: L_Total = L_CE + lambda * L_RIL
        """
        super().__init__()
        self.lambda_ril = lambda_ril
        self.ce_loss = nn.CrossEntropyLoss()
        self.ril_loss = RegionalIndependenceLoss(margin=margin, gamma_feat=gamma_feat)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        V_seq: torch.Tensor = None,
        A_seq: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: (B, 2) classification logits
            targets: (B,) class labels (0: Real, 1: Fake)
            V_seq: Optional (B, T, M, C)
            A_seq: Optional (B, T, M, H, W)
        """
        l_ce = self.ce_loss(logits, targets)
        
        if V_seq is not None and A_seq is not None and self.lambda_ril > 0:
            l_ril, l_spatial, l_feat = self.ril_loss(V_seq, A_seq)
            l_total = l_ce + self.lambda_ril * l_ril
            metrics = {
                "loss_total": l_total.item(),
                "loss_ce": l_ce.item(),
                "loss_ril": l_ril.item(),
                "loss_spatial": l_spatial.item(),
                "loss_feat": l_feat.item()
            }
        else:
            l_total = l_ce
            metrics = {
                "loss_total": l_total.item(),
                "loss_ce": l_ce.item(),
                "loss_ril": 0.0,
                "loss_spatial": 0.0,
                "loss_feat": 0.0
            }
            
        return l_total, metrics
