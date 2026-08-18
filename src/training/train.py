"""
Main 5-Fold Cross-Validation Training Script for Spatiotemporal Deepfake Detection.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from typing import List, Dict

from ..config import ModelConfig, LossConfig, TrainingConfig
from ..models.spatiotemporal_net import SpatiotemporalDeepfakeDetector
from ..losses.loss import TotalDetectionLoss
from ..dataset.video_dataset import VideoSequenceDataset
from ..dataset.split_utils import partition_identity_disjoint_folds
from .trainer import DeepfakeTrainer


def run_5fold_cross_validation(
    samples: List[Dict],
    model_cfg: ModelConfig = ModelConfig(),
    loss_cfg: LossConfig = LossConfig(),
    train_cfg: TrainingConfig = TrainingConfig()
) -> Dict[str, Tuple[float, float]]:
    """
    Executes 5-fold identity-disjoint cross-validation and computes Mean +- Std Dev metrics.
    """
    # 1. Subject-Disjoint Partitioning
    fold_splits = partition_identity_disjoint_folds(
        samples=samples,
        k_folds=train_cfg.k_folds,
        seed=train_cfg.seed
    )
    
    fold_results = []
    
    for fold_idx, (train_samples, val_samples) in enumerate(fold_splits):
        print(f"\n{'='*25} Starting Fold {fold_idx+1}/{train_cfg.k_folds} {'='*25}")
        print(f"Train samples: {len(train_samples)} | Validation samples: {len(val_samples)}")
        
        # Build DataLoaders
        train_dataset = VideoSequenceDataset(
            samples=train_samples,
            sequence_length=model_cfg.sequence_length,
            is_train=True,
            use_freq_aug=True
        )
        val_dataset = VideoSequenceDataset(
            samples=val_samples,
            sequence_length=model_cfg.sequence_length,
            is_train=False,
            use_freq_aug=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=train_cfg.num_workers,
            pin_memory=True if train_cfg.device == "cuda" else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=True if train_cfg.device == "cuda" else False
        )
        
        # Instantiate Model, Loss, Optimizer
        model = SpatiotemporalDeepfakeDetector(
            backbone_name=model_cfg.backbone,
            pretrained=model_cfg.pretrained,
            num_heads=model_cfg.num_attention_heads,
            texture_channels=model_cfg.texture_channels,
            semantic_channels=model_cfg.semantic_channels,
            bottleneck_dim=model_cfg.lstm_input_dim,
            lstm_hidden_dim=model_cfg.lstm_hidden_dim,
            lstm_layers=model_cfg.lstm_layers,
            dropout=model_cfg.dropout,
            num_classes=model_cfg.num_classes
        )
        
        criterion = TotalDetectionLoss(
            lambda_ril=loss_cfg.lambda_ril,
            margin=loss_cfg.ril_margin,
            gamma_feat=loss_cfg.gamma_feat
        )
        
        optimizer = Adam(
            model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay
        )
        
        scheduler = StepLR(
            optimizer,
            step_size=train_cfg.lr_decay_step,
            gamma=train_cfg.lr_decay_gamma
        )
        
        trainer = DeepfakeTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=train_cfg.device,
            agda_prob=loss_cfg.agda_prob,
            checkpoint_dir=train_cfg.checkpoint_dir
        )
        
        # Train fold
        best_fold_metrics = trainer.train_and_validate(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=train_cfg.epochs,
            fold_idx=fold_idx
        )
        fold_results.append(best_fold_metrics)
        
    # Aggregate across 5 folds
    aggregated_metrics = {}
    metric_keys = ["acc", "auc", "precision", "recall", "f1", "fpr", "fnr"]
    
    print("\n" + "="*60)
    print("5-FOLD CROSS-VALIDATION FINAL SUMMARY (Mean +- Std Dev)")
    print("="*60)
    
    for key in metric_keys:
        values = [res[key] for res in fold_results if key in res]
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            aggregated_metrics[key] = (mean_val, std_val)
            print(f"{key.upper():<12}: {mean_val:.2f}% +- {std_val:.2f}%")
            
    return aggregated_metrics
