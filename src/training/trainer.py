"""
Trainer Module for Spatiotemporal Deepfake Detection.
Handles training loops, AGDA regularization, multi-loss optimization, and validation.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from typing import Dict, Tuple, Optional

from ..losses.loss import TotalDetectionLoss
from ..evaluation.evaluate import evaluate_model


class DeepfakeTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: TotalDetectionLoss,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        agda_prob: float = 0.5,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Trainer class managing training epochs and validation cycles.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.agda_prob = agda_prob
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _apply_agda(self, clips: torch.Tensor, A_seq: torch.Tensor) -> torch.Tensor:
        """
        Applies Attention-Guided Data Augmentation (AGDA).
        Blurs or masks the highest activation region to prevent single-cue over-reliance.
        """
        if torch.rand(1).item() > self.agda_prob:
            return clips
            
        B, T, C, H, W = clips.shape
        # Average attention across heads and time: (B, H, W)
        mean_attn = torch.mean(A_seq, dim=(1, 2))  # (B, H_a, W_a)
        
        # Interpolate attention map to frame resolution (H, W)
        attn_upsampled = nn.functional.interpolate(
            mean_attn.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)  # (B, H, W)
        
        # Threshold top 20% activation mask
        threshold = torch.quantile(attn_upsampled.view(B, -1), 0.8, dim=1, keepdim=True)
        mask = (attn_upsampled > threshold.view(B, 1, 1)).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
        
        # Soft blur masking on dominant attention zone
        blurred_clips = nn.functional.avg_pool3d(
            clips.permute(0, 2, 1, 3, 4), kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)
        ).permute(0, 2, 1, 3, 4)
        
        aug_clips = torch.where(mask, blurred_clips, clips)
        return aug_clips

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Dict[str, float]:
        """Runs one full training epoch."""
        self.model.train()
        total_loss = 0.0
        ce_loss_sum = 0.0
        ril_loss_sum = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx+1} [Train]", leave=False)
        for clips, labels in pbar:
            clips = clips.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass extracting predictions and attention tensors
            logits, V_seq, A_seq = self.model(clips, return_attention=True)
            
            # Optional AGDA forward pass on masked inputs
            if self.agda_prob > 0 and A_seq is not None:
                with torch.no_grad():
                    aug_clips = self._apply_agda(clips, A_seq)
                logits_aug, V_seq_aug, A_seq_aug = self.model(aug_clips, return_attention=True)
                loss_clean, metrics = self.criterion(logits, labels, V_seq, A_seq)
                loss_aug, _ = self.criterion(logits_aug, labels, V_seq_aug, A_seq_aug)
                loss = 0.5 * (loss_clean + loss_aug)
            else:
                loss, metrics = self.criterion(logits, labels, V_seq, A_seq)
                
            loss.backward()
            
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += metrics["loss_total"]
            ce_loss_sum += metrics["loss_ce"]
            ril_loss_sum += metrics["loss_ril"]
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "CE": f"{metrics['loss_ce']:.4f}", "RIL": f"{metrics['loss_ril']:.4f}"})
            
        if self.scheduler is not None:
            self.scheduler.step()
            
        return {
            "train_loss": total_loss / max(num_batches, 1),
            "train_ce": ce_loss_sum / max(num_batches, 1),
            "train_ril": ril_loss_sum / max(num_batches, 1)
        }

    def train_and_validate(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        fold_idx: int = 0
    ) -> Dict[str, float]:
        """
        Executes full multi-epoch training and validation loop for a given fold.
        """
        best_auc = 0.0
        best_metrics = {}
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = evaluate_model(self.model, val_loader, device=self.device)
            
            print(f"Fold {fold_idx+1} | Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['train_loss']:.4f} | Val ACC: {val_metrics['acc']:.2f}% | Val AUC: {val_metrics['auc']:.2f}% | Val F1: {val_metrics['f1']:.2f}%")
            
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_metrics = val_metrics
                # Save best fold checkpoint
                ckpt_path = os.path.join(self.checkpoint_dir, f"best_model_fold_{fold_idx+1}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
                
        return best_metrics
