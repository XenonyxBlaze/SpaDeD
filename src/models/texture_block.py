"""
Texture Enhancement Block (TEB).
Extracts shallow high-frequency microscopic artifacts and residual noise patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureEnhancementBlock(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 256, target_hw: int = 8):
        """
        Args:
            in_channels: Input channels from shallow backbone stage (default: 64 from layer 1)
            out_channels: Output texture feature channels C_tex (default: 256)
            target_hw: Target spatial height and width to match Stage 4 (default: 8x8 for 256x256 input)
        """
        super().__init__()
        
        # Shallow residual branch
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Spatial downsampling & channel projection to match Stage 4 output (8x8)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),          # 32 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=3, stride=2, padding=1, bias=False), # 16 -> 8
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_low: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_low: (B, 64, 64, 64) shallow feature map
        Returns:
            F_tex: (B, 256, 8, 8) texture feature map
        """
        residual = x_low
        out = self.conv1x1(x_low)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv3x3(out)
        out = self.bn2(out)
        
        # Residual fusion
        out = self.relu(out + residual)
        
        # Downsample to target spatial resolution (H=8, W=8)
        f_tex = self.downsample(out)
        return f_tex
