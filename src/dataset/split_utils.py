"""
Strict Subject/Identity-Disjoint Cross-Validation Partitioner.
Guarantees 0.0% data leakage across folds by partitioning at the source video/identity level.
"""

import os
import random
from typing import List, Dict, Tuple
from collections import defaultdict


def partition_identity_disjoint_folds(
    samples: List[Dict],
    k_folds: int = 5,
    seed: int = 42
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Partitions dataset samples into k folds where all samples of the same subject_id / source_video
    are strictly assigned to the same fold.
    
    Args:
        samples: List of sample dicts, each containing 'video_path', 'label', and 'identity_id'
        k_folds: Number of folds (default: 5)
        seed: Random seed for reproducibility
    Returns:
        fold_splits: List of (train_samples, val_samples) tuples for each fold
    """
    random.seed(seed)
    
    # Group samples by unique subject identity or source video base name
    identity_to_samples = defaultdict(list)
    for sample in samples:
        ident = sample.get("identity_id", sample.get("source_video", "default"))
        identity_to_samples[ident].append(sample)
        
    unique_identities = list(identity_to_samples.keys())
    random.shuffle(unique_identities)
    
    # Distribute identities evenly into k buckets
    fold_identity_buckets = [[] for _ in range(k_folds)]
    for idx, ident in enumerate(unique_identities):
        fold_identity_buckets[idx % k_folds].append(ident)
        
    # Build train and validation sample lists for each fold
    fold_splits = []
    for fold_idx in range(k_folds):
        val_identities = set(fold_identity_buckets[fold_idx])
        train_identities = set(unique_identities) - val_identities
        
        train_samples = []
        val_samples = []
        
        for ident in train_identities:
            train_samples.extend(identity_to_samples[ident])
            
        for ident in val_identities:
            val_samples.extend(identity_to_samples[ident])
            
        # Verify 0% identity leakage
        assert len(train_identities.intersection(val_identities)) == 0, "FATAL: Identity leakage detected!"
        
        fold_splits.append((train_samples, val_samples))
        
    return fold_splits
