"""
Configuration and Hyperparameter Settings for Spatiotemporal Deepfake Detection.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os

@dataclass
class ModelConfig:
    backbone: str = "resnext50_32x4d"
    pretrained: bool = True
    num_classes: int = 2
    num_attention_heads: int = 4
    texture_channels: int = 256
    semantic_channels: int = 2048  # ResNeXt50 stage 4 channels
    fused_channels: int = 2304     # 2048 + 256
    lstm_input_dim: int = 512
    lstm_hidden_dim: int = 256     # Per direction (bidirectional -> 512 total)
    lstm_layers: int = 2
    dropout: float = 0.3
    sequence_length: int = 20      # T frames per video clip
    image_size: Tuple[int, int] = (256, 256)

@dataclass
class LossConfig:
    lambda_ril: float = 0.5        # Weight for Regional Independence Loss
    gamma_feat: float = 1.0        # Weight for feature orthogonality within LRIL
    ril_margin: float = 0.2        # Margin m for cosine similarity penalty
    agda_prob: float = 0.5         # Attention Guided Data Augmentation probability
    agda_type: str = "soft"        # 'soft' blurring or 'hard' erasing

@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_decay_step: int = 5
    lr_decay_gamma: float = 0.1
    epochs: int = 15
    k_folds: int = 5
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results_output"

@dataclass
class DataConfig:
    data_root: str = "data"
    ffpp_root: str = "data/FaceForensics++"
    df40_root: str = "data/DF40"
    celebdf_root: str = "data/Celeb-DF-v2"
    categories: List[str] = field(default_factory=lambda: ["FS", "FR", "EFS", "FE"])
    train_category: str = "FS"     # Primary training family for cross-forgery test
