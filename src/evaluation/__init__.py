from .metrics import compute_classification_metrics
from .statistical_tests import perform_paired_statistical_analysis, apply_multiple_comparison_corrections

__all__ = [
    "compute_classification_metrics",
    "perform_paired_statistical_analysis",
    "apply_multiple_comparison_corrections"
]

def get_evaluator():
    from .evaluate import evaluate_model
    return evaluate_model
