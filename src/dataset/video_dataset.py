"""
Video Sequence Dataset for Spatiotemporal Deepfake Detection.
Extracts consecutive or uniformly spaced T=20 frames from video files or pre-extracted frame folders.
"""

import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Dict, Optional, Callable

from .transforms import get_default_transforms, FrequencyPreservingAugmentation


class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        sequence_length: int = 20,
        is_train: bool = True,
        use_freq_aug: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            samples: List of dicts with keys: 'path' (video file or frame dir), 'label' (0 or 1), 'identity_id'
            sequence_length: Number of frames per clip T (default: 20)
            is_train: Training mode
            use_freq_aug: Enable Fourier frequency-preserving augmentation
        """
        self.samples = samples
        self.sequence_length = sequence_length
        self.is_train = is_train
        self.use_freq_aug = use_freq_aug and is_train
        self.freq_aug = FrequencyPreservingAugmentation() if self.use_freq_aug else None
        self.transform = transform if transform is not None else get_default_transforms(is_train=is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """Extracts T frames from an MP4/AVI video file."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        if total_frames <= 0:
            # Fallback black frames if video cannot be opened
            return [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(self.sequence_length)]
            
        # Select indices
        if total_frames >= self.sequence_length:
            if self.is_train:
                # Random contiguous clip
                max_start = total_frames - self.sequence_length
                start_idx = np.random.randint(0, max_start + 1)
                indices = list(range(start_idx, start_idx + self.sequence_length))
            else:
                # Uniformly spaced sampling
                indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        else:
            # Replicate frames if video is shorter than T
            indices = [i % total_frames for i in range(self.sequence_length)]
            
        current_frame = 0
        frame_dict = {}
        target_set = set(indices)
        
        while cap.isOpened() and len(frame_dict) < len(target_set):
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in target_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_dict[current_frame] = frame_rgb
            current_frame += 1
        cap.release()
        
        for idx in indices:
            if idx in frame_dict:
                frames.append(frame_dict[idx])
            elif len(frame_dict) > 0:
                frames.append(list(frame_dict.values())[0])
            else:
                frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
                
        return frames

    def _load_frames_from_dir(self, dir_path: str) -> List[np.ndarray]:
        """Loads T frames from a folder of images."""
        img_paths = sorted(glob.glob(os.path.join(dir_path, "*.png")) + glob.glob(os.path.join(dir_path, "*.jpg")))
        total_frames = len(img_paths)
        
        if total_frames == 0:
            return [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(self.sequence_length)]
            
        if total_frames >= self.sequence_length:
            if self.is_train:
                start_idx = np.random.randint(0, total_frames - self.sequence_length + 1)
                selected_paths = img_paths[start_idx : start_idx + self.sequence_length]
            else:
                indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
                selected_paths = [img_paths[i] for i in indices]
        else:
            selected_paths = [img_paths[i % total_frames] for i in range(self.sequence_length)]
            
        frames = []
        for p in selected_paths:
            img = Image.open(p).convert("RGB")
            frames.append(np.array(img))
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        path = sample["path"]
        label = sample["label"]
        
        if os.path.isdir(path):
            raw_frames = self._load_frames_from_dir(path)
        else:
            raw_frames = self._load_frames_from_video(path)
            
        tensor_frames = []
        for f_np in raw_frames:
            if self.use_freq_aug and self.freq_aug is not None and np.random.rand() > 0.5:
                f_np = self.freq_aug(f_np)
            pil_img = Image.fromarray(f_np)
            t_frame = self.transform(pil_img)  # (3, 256, 256)
            tensor_frames.append(t_frame)
            
        # Stack to temporal tensor: (T, 3, 256, 256)
        clip_tensor = torch.stack(tensor_frames, dim=0)
        return clip_tensor, label
