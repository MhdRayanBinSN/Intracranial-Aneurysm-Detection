"""
Evaluation metrics for multi-label classification.
Implements the competition metric: Weighted Multilabel AUC.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Optional, Tuple
import warnings


# Anatomical location names for reporting
LOCATION_NAMES = [
    'L Infraclinoid ICA',
    'R Infraclinoid ICA',
    'L Supraclinoid ICA',
    'R Supraclinoid ICA',
    'L Middle Cerebral',
    'R Middle Cerebral',
    'L Anterior Cerebral',
    'R Anterior Cerebral',
    'L Posterior Comm',
    'R Posterior Comm',
    'Basilar Tip',
    'Other Posterior',
    'Overall Presence',
]


class MultilabelAUC:
    """
    Computes per-class and weighted multilabel AUC.
    
    This is the competition metric for RSNA Intracranial Aneurysm Detection.
    """
    
    def __init__(
        self,
        num_classes: int = 14,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Args:
            num_classes: Number of classes
            weights: Optional weights for each class (for weighted average)
        """
        self.num_classes = num_classes
        self.weights = weights if weights is not None else np.ones(num_classes)
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets."""
        self.all_targets = []
        self.all_probs = []
    
    def update(
        self,
        targets: torch.Tensor,
        probs: torch.Tensor,
    ):
        """
        Update with batch of predictions.
        
        Args:
            targets: Ground truth labels (B, C)
            probs: Predicted probabilities (B, C)
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        
        self.all_targets.append(targets)
        self.all_probs.append(probs)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute AUC metrics.
        
        Returns:
            Dictionary with per-class AUC and weighted average
        """
        if len(self.all_targets) == 0:
            return {'weighted_auc': 0.0}
        
        targets = np.vstack(self.all_targets)
        probs = np.vstack(self.all_probs)
        
        results = {}
        valid_aucs = []
        valid_weights = []
        
        for i in range(self.num_classes):
            class_targets = targets[:, i]
            class_probs = probs[:, i]
            
            # Skip if only one class present
            if len(np.unique(class_targets)) < 2:
                results[f'auc_class_{i}'] = np.nan
                continue
            
            try:
                auc = roc_auc_score(class_targets, class_probs)
                results[f'auc_class_{i}'] = auc
                valid_aucs.append(auc)
                valid_weights.append(self.weights[i])
            except Exception as e:
                results[f'auc_class_{i}'] = np.nan
        
        # Compute weighted average
        if len(valid_aucs) > 0:
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()
            weighted_auc = np.average(valid_aucs, weights=valid_weights)
        else:
            weighted_auc = 0.0
        
        results['weighted_auc'] = weighted_auc
        results['mean_auc'] = np.nanmean(list(results.values())[:-1]) if valid_aucs else 0.0
        
        return results
    
    def compute_detailed(self) -> Dict[str, float]:
        """
        Compute detailed metrics with location names.
        
        Returns:
            Dictionary with named metrics
        """
        basic_results = self.compute()
        
        detailed = {'weighted_auc': basic_results['weighted_auc']}
        
        for i, name in enumerate(LOCATION_NAMES):
            key = f'auc_class_{i}'
            if key in basic_results:
                detailed[f'auc_{name}'] = basic_results[key]
        
        return detailed


class MultilabelMetrics:
    """
    Comprehensive metrics for multi-label classification.
    """
    
    def __init__(self, num_classes: int = 14, threshold: float = 0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.all_targets = []
        self.all_probs = []
        self.all_preds = []
    
    def update(
        self,
        targets: torch.Tensor,
        probs: torch.Tensor,
    ):
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        
        preds = (probs > self.threshold).astype(int)
        
        self.all_targets.append(targets)
        self.all_probs.append(probs)
        self.all_preds.append(preds)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if len(self.all_targets) == 0:
            return {}
        
        targets = np.vstack(self.all_targets)
        probs = np.vstack(self.all_probs)
        preds = np.vstack(self.all_preds)
        
        results = {}
        
        # Per-class metrics
        for i in range(self.num_classes):
            class_targets = targets[:, i]
            class_probs = probs[:, i]
            class_preds = preds[:, i]
            
            # Accuracy
            results[f'acc_class_{i}'] = (class_targets == class_preds).mean()
            
            # Precision, Recall, F1
            tp = ((class_targets == 1) & (class_preds == 1)).sum()
            fp = ((class_targets == 0) & (class_preds == 1)).sum()
            fn = ((class_targets == 1) & (class_preds == 0)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            results[f'precision_class_{i}'] = precision
            results[f'recall_class_{i}'] = recall
            results[f'f1_class_{i}'] = f1
            
            # AUC
            if len(np.unique(class_targets)) >= 2:
                try:
                    results[f'auc_class_{i}'] = roc_auc_score(class_targets, class_probs)
                except:
                    results[f'auc_class_{i}'] = np.nan
        
        # Average metrics
        results['mean_accuracy'] = np.mean([results[f'acc_class_{i}'] for i in range(self.num_classes)])
        results['mean_precision'] = np.mean([results[f'precision_class_{i}'] for i in range(self.num_classes)])
        results['mean_recall'] = np.mean([results[f'recall_class_{i}'] for i in range(self.num_classes)])
        results['mean_f1'] = np.mean([results[f'f1_class_{i}'] for i in range(self.num_classes)])
        
        valid_aucs = [results[f'auc_class_{i}'] for i in range(self.num_classes) 
                      if not np.isnan(results.get(f'auc_class_{i}', np.nan))]
        results['mean_auc'] = np.mean(valid_aucs) if valid_aucs else 0.0
        
        # Overall accuracy (exact match)
        results['exact_match_accuracy'] = (targets == preds).all(axis=1).mean()
        
        return results


def compute_competition_metric(
    targets: np.ndarray,
    probs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """
    Compute the competition metric: Weighted Multilabel AUC.
    
    Args:
        targets: Ground truth (N, 14)
        probs: Predictions (N, 14)
        weights: Optional class weights (14,)
        
    Returns:
        Weighted multilabel AUC score
    """
    num_classes = targets.shape[1]
    
    if weights is None:
        weights = np.ones(num_classes)
    
    aucs = []
    valid_weights = []
    
    for i in range(num_classes):
        if len(np.unique(targets[:, i])) < 2:
            continue
        
        try:
            auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(auc)
            valid_weights.append(weights[i])
        except:
            continue
    
    if len(aucs) == 0:
        return 0.0
    
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    
    return np.average(aucs, weights=valid_weights)


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics...")
    
    # Create dummy data
    targets = torch.randint(0, 2, (100, 14)).float()
    probs = torch.sigmoid(torch.randn(100, 14))
    
    # Test MultilabelAUC
    auc_metric = MultilabelAUC(num_classes=14)
    auc_metric.update(targets, probs)
    results = auc_metric.compute()
    print(f"Weighted AUC: {results['weighted_auc']:.4f}")
    
    # Test detailed
    detailed = auc_metric.compute_detailed()
    print(f"Detailed results: {len(detailed)} metrics")
    
    # Test MultilabelMetrics
    metrics = MultilabelMetrics(num_classes=14)
    metrics.update(targets, probs)
    all_metrics = metrics.compute()
    print(f"Mean F1: {all_metrics['mean_f1']:.4f}")
    print(f"Mean AUC: {all_metrics['mean_auc']:.4f}")
    
    print("\nAll metrics working correctly!")
