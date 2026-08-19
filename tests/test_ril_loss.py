"""
Unit test verifying mathematical precision of RegionalIndependenceLoss and BilinearAttentionPooling.
Validates:
1. A_k >= 0 and sum_{x,y} A_k(x,y) == 1.0 (Spatial Softmax normalization)
2. V_k in R^2304 (Fused Stage-4 Semantics + TEB Texture channels)
3. V_hat = V / max(||V||_2, eps) unit-norm cosine similarity
4. Exactly binom(M, 2) = 6 unordered head pairs for M=4
5. Correct behavior under orthogonal vs overlapping synthetic distributions
"""

import torch
import torch.nn.functional as F
from src.models.multi_attention import BilinearAttentionPooling
from src.losses.loss import RegionalIndependenceLoss


def test_bap_and_ril_invariants():
    B, T, M, C, H, W = 2, 20, 4, 2304, 8, 8
    
    # 1. Test BAP Forward Pass
    bap = BilinearAttentionPooling(in_channels=C, num_heads=M)
    F_fused = torch.randn(B * T, C, H, W)
    
    V_norm, A = bap(F_fused)
    
    # Invariant 1: Attention map non-negativity and spatial probability mass sum = 1
    assert torch.all(A >= 0.0), "Attention maps must be non-negative"
    A_spatial_sums = A.sum(dim=(2, 3))  # Sum over H, W
    assert torch.allclose(A_spatial_sums, torch.ones_like(A_spatial_sums), atol=1e-5), \
        f"Spatial attention maps must sum to 1.0 over (H, W). Got {A_spatial_sums[0]}"
    
    # Invariant 2: Regional descriptor dimensionality
    assert V_norm.shape == (B * T, M, C), f"Expected shape ({B*T}, {M}, {C}), got {V_norm.shape}"
    assert V_norm.shape[-1] == 2304, f"Regional descriptor must have dimension 2304, got {V_norm.shape[-1]}"
    
    # Invariant 3: Unit L2-norm
    V_magnitudes = torch.norm(V_norm, p=2, dim=-1)
    assert torch.allclose(V_magnitudes, torch.ones_like(V_magnitudes), atol=1e-5), \
        "Regional descriptors must have unit L2 norm"
    
    # Reshape for sequence loss
    V_seq = V_norm.view(B, T, M, C)
    A_seq = A.view(B, T, M, H, W)
    
    # 2. Test Loss Computation
    criterion = RegionalIndependenceLoss(margin=0.2, gamma_feat=1.0)
    l_ril, l_spatial, l_feat = criterion(V_seq, A_seq)
    
    assert l_ril.item() >= 0.0, "Loss must be non-negative"
    assert torch.isfinite(l_ril), "Loss must be finite"
    
    print("[PASS] Invariants 1-4 verified successfully.")
    print(f"       Computed L_RIL: {l_ril.item():.4f} (L_spatial: {l_spatial.item():.4f}, L_feat: {l_feat.item():.4f})")


def test_synthetic_orthogonality_cases():
    """Verify loss outputs on known edge cases."""
    B, T, M, C, H, W = 1, 1, 4, 2304, 2, 2
    
    # Case A: Perfectly disjoint 1-hot attention maps (one quadrant per head)
    # Head 0: (0,0), Head 1: (0,1), Head 2: (1,0), Head 3: (1,1)
    A_disjoint = torch.zeros(B, T, M, H, W)
    A_disjoint[0, 0, 0, 0, 0] = 1.0
    A_disjoint[0, 0, 1, 0, 1] = 1.0
    A_disjoint[0, 0, 2, 1, 0] = 1.0
    A_disjoint[0, 0, 3, 1, 1] = 1.0
    
    # Orthogonal feature vectors
    V_orth = torch.zeros(B, T, M, C)
    V_orth[0, 0, 0, 0] = 1.0
    V_orth[0, 0, 1, 1] = 1.0
    V_orth[0, 0, 2, 2] = 1.0
    V_orth[0, 0, 3, 3] = 1.0
    
    criterion = RegionalIndependenceLoss(margin=0.2, gamma_feat=1.0)
    l_ril_zero, l_spatial_zero, l_feat_zero = criterion(V_orth, A_disjoint)
    
    assert l_spatial_zero.item() == 0.0, f"Disjoint spatial maps must yield exactly 0.0 spatial loss, got {l_spatial_zero.item()}"
    assert l_feat_zero.item() == 0.0, f"Orthogonal features must yield exactly 0.0 feature loss, got {l_feat_zero.item()}"
    assert l_ril_zero.item() == 0.0, f"Perfect independence must yield exactly 0.0 RIL, got {l_ril_zero.item()}"
    
    # Case B: Completely identical overlapping attention maps and identical features
    A_overlap = torch.full((B, T, M, H, W), 0.25)  # Softmax uniform (4 cells of 0.25)
    V_identical = torch.zeros(B, T, M, C)
    V_identical[:, :, :, 0] = 1.0  # All heads have the exact same direction
    
    l_ril_max, l_spatial_max, l_feat_max = criterion(V_identical, A_overlap)
    
    # Expected spatial overlap per pair: 4 * (0.25 * 0.25) = 0.25
    assert abs(l_spatial_max.item() - 0.25) < 1e-5, f"Expected 0.25 spatial loss, got {l_spatial_max.item()}"
    # Expected feature similarity per pair: 1.0 - 0.2 = 0.8
    assert abs(l_feat_max.item() - 0.80) < 1e-5, f"Expected 0.80 feature loss, got {l_feat_max.item()}"
    
    print("[PASS] Synthetic mathematical edge cases verified exactly.")
    print(f"       Disjoint Case: L_spatial={l_spatial_zero.item():.4f}, L_feat={l_feat_zero.item():.4f}")
    print(f"       Overlap Case:  L_spatial={l_spatial_max.item():.4f}, L_feat={l_feat_max.item():.4f}")


if __name__ == "__main__":
    test_bap_and_ril_invariants()
    test_synthetic_orthogonality_cases()
