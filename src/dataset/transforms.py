"""
Frequency-Preserving Fourier Data Augmentation and Standard Transforms.
Protects high-frequency forensic cues from conventional degradation.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from typing import Tuple


class FrequencyPreservingAugmentation:
    def __init__(self, low_pass_radius: int = 30, alpha_range: Tuple[float, float] = (0.3, 0.7)):
        """
        Mixes low-frequency Fourier components while leaving high frequencies intact.
        """
        self.radius = low_pass_radius
        self.alpha_range = alpha_range

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        """
        Args:
            img_np: (H, W, 3) uint8 numpy array
        Returns:
            aug_np: (H, W, 3) uint8 numpy array
        """
        H, W, C = img_np.shape
        cy, cx = H // 2, W // 2
        
        # Create circular low-pass binary mask
        y, x = np.ogrid[:H, :W]
        mask = ((y - cy) ** 2 + (x - cx) ** 2) <= (self.radius ** 2)
        mask = mask[:, :, np.newaxis]
        
        # Synthetic low-frequency noise image
        noise = np.random.randint(0, 256, (H, W, C), dtype=np.uint8)
        alpha = np.random.uniform(*self.alpha_range)
        
        # Fourier Transform per channel
        fft_img = np.fft.fftshift(np.fft.fft2(img_np, axes=(0, 1)), axes=(0, 1))
        fft_noise = np.fft.fftshift(np.fft.fft2(noise, axes=(0, 1)), axes=(0, 1))
        
        # Blend in low frequencies only
        fft_fused = (1.0 - mask * alpha) * fft_img + (mask * alpha) * fft_noise
        
        # Inverse Fourier Transform
        ifft_fused = np.fft.ifft2(np.fft.ifftshift(fft_fused, axes=(0, 1)), axes=(0, 1))
        aug_np = np.clip(np.real(ifft_fused), 0, 255).astype(np.uint8)
        return aug_np


def get_default_transforms(is_train: bool = True, image_size: Tuple[int, int] = (256, 256)):
    """
    Standard normalization and tensor conversion.
    """
    if is_train:
        return T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
