"""
Static Image Dataset for Spatiotemporal Evaluation (EFS and FE benchmarks).
Implements the temporal replication protocol: repeats static image across T=20 time steps.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional, Callable

from .transforms import get_default_transforms, FrequencyPreservingAugmentation


class StaticImageSequenceDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        sequence_length: int = 20,
        is_train: bool = False,
        use_freq_aug: bool = False,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            samples: List of dicts with keys: 'path' (image file), 'label' (0 or 1), 'generator'
            sequence_length: Temporal replication length T (default: 20)
        """
        self.samples = samples
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.use_freq_aug = use_freq_aug and is_train
        self.freq_aug = FrequencyPreservingAugmentation() if self.use_freq_aug else None
        self.transform = transform if transform is not None else get_default_transforms(is_train=is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img_path = sample["path"]
        label = sample["label"]
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback blank image
            img = Image.new("RGB", (256, 256), color=(0, 0, 0))
            
        img_np = np.array(img)
        if self.use_freq_aug and self.freq_aug is not None and np.random.rand() > 0.5:
            img_np = self.freq_aug(img_np)
            img = Image.fromarray(img_np)
            
        # Transform single static image: (3, 256, 256)
        tensor_img = self.transform(img)
        
        # Temporal Replication Protocol: Repeat single frame T=20 times -> (T, 3, 256, 256)
        clip_tensor = tensor_img.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        return clip_tensor, label
