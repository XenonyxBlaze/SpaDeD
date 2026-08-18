"""
Evaluation Runner over PyTorch DataLoader.
Computes complete classification metrics and confusion matrix.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple

from .metrics import compute_classification_metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluates the model over the provided DataLoader.
    
    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader yielding (clips, labels)
        device: 'cuda' or 'cpu'
    Returns:
        metrics: Dict with ACC, AUC, Precision, Recall, F1, FPR, FNR
    """
    model.eval()
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for clips, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            clips = clips.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _, _ = model(clips, return_attention=False)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of Fake class
            
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            
    metrics = compute_classification_metrics(all_targets, all_probs)
    return metrics
