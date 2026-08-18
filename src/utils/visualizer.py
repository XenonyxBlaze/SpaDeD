"""
Attention Map and Heatmap Visualization Utilities.
Plots the M=4 regional attention maps overlaid on face images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from typing import List, Optional


def overlay_attention_map(image_np: np.ndarray, attention_map_2d: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a single 2D attention map onto an RGB image.
    
    Args:
        image_np: (H, W, 3) uint8 image
        attention_map_2d: (H_a, W_a) float attention weights
        alpha: Overlay blend weight
    Returns:
        overlaid_np: (H, W, 3) uint8 RGB image
    """
    H, W, _ = image_np.shape
    # Resize attention map to image dimensions
    heatmap = cv2.resize(attention_map_2d, (W, H))
    # Normalize to [0, 255]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply JET colormap
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlaid = np.uint8(alpha * color_map + (1 - alpha) * image_np)
    return overlaid


def plot_multihead_attention(
    image_np: np.ndarray,
    attention_maps_4d: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plots the original image and M=4 attention heatmaps side by side.
    
    Args:
        image_np: (H, W, 3) uint8 RGB image
        attention_maps_4d: (M, H_a, W_a) attention map tensor
        save_path: Optional file path to save figure
    """
    M = attention_maps_4d.shape[0]
    attn_np = attention_maps_4d.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, M + 1, figsize=(18, 4))
    
    axes[0].imshow(image_np)
    axes[0].set_title("Input Frame", fontsize=12)
    axes[0].axis("off")
    
    region_labels = ["Head 1 (Eyes/Brow)", "Head 2 (Nose/Cheek)", "Head 3 (Mouth/Chin)", "Head 4 (Boundary)"]
    
    for i in range(M):
        overlaid = overlay_attention_map(image_np, attn_np[i])
        axes[i + 1].imshow(overlaid)
        title = region_labels[i] if i < len(region_labels) else f"Head {i+1}"
        axes[i + 1].set_title(title, fontsize=12)
        axes[i + 1].axis("off")
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Attention visualization saved to: {save_path}")
    plt.close()
