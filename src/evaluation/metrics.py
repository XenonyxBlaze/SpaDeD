"""
Evaluation Metrics for Deepfake Detection.
Computes Accuracy, ROC-AUC, Precision, Recall, F1-Score, FPR, and FNR.
"""

from typing import Dict, List, Tuple


def compute_classification_metrics(y_true: List[int], y_pred_probs: List[float], threshold: float = 0.5) -> Dict[str, float]:
    """
    Args:
        y_true: List of ground-truth binary labels (0: Real, 1: Fake)
        y_pred_probs: List of predicted probabilities for positive class (Fake)
        threshold: Decision threshold for classification
    Returns:
        metrics: Dictionary containing ACC, AUC, Precision, Recall, F1, FPR, FNR
    """
    try:
        import numpy as np
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
        
        y_true_np = np.array(y_true)
        y_pred_probs_np = np.array(y_pred_probs)
        y_pred_np = (y_pred_probs_np >= threshold).astype(int)
        
        acc = accuracy_score(y_true_np, y_pred_np) * 100.0
        
        if len(np.unique(y_true_np)) > 1:
            auc = roc_auc_score(y_true_np, y_pred_probs_np) * 100.0
        else:
            auc = 50.0
            
        prec = precision_score(y_true_np, y_pred_np, zero_division=0) * 100.0
        rec = recall_score(y_true_np, y_pred_np, zero_division=0) * 100.0
        f1 = f1_score(y_true_np, y_pred_np, zero_division=0) * 100.0
        
        tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
        fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) > 0 else 0.0
        fnr = (fn / (fn + tp)) * 100.0 if (fn + tp) > 0 else 0.0
        
    except ImportError:
        # Pure Python fallback calculations
        n = len(y_true)
        y_pred = [1 if p >= threshold else 0 for p in y_pred_probs]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        
        acc = (tp + tn) / max(n, 1) * 100.0
        prec = tp / max(tp + fp, 1) * 100.0
        rec = tp / max(tp + fn, 1) * 100.0
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8)
        fpr = fp / max(fp + tn, 1) * 100.0
        fnr = fn / max(fn + tp, 1) * 100.0
        auc = 0.5 * (rec + (100.0 - fpr))
        
    return {
        "acc": float(acc),
        "auc": float(auc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr)
    }
