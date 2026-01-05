"""
Training module initialization.

Contains:
- Classification: Trainer, FocalLoss, etc.
- Segmentation: DiceLoss, DiceBCELoss
- Multi-task: MultiTaskLoss, MultiTaskTrainer
"""

from .losses import (
    WeightedBCELoss,
    FocalLoss,
    AsymmetricLoss,
    CombinedLoss,
    DiceLoss,
    DiceBCELoss,
    MultiTaskLoss,
    get_loss_function,
)
from .metrics import (
    MultilabelAUC,
    MultilabelMetrics,
    compute_competition_metric,
    LOCATION_NAMES,
)
from .trainer import Trainer
from .multitask_trainer import MultiTaskTrainer

__all__ = [
    # Classification Losses
    'WeightedBCELoss',
    'FocalLoss',
    'AsymmetricLoss',
    'CombinedLoss',
    'get_loss_function',
    # Segmentation Losses (NEW)
    'DiceLoss',
    'DiceBCELoss',
    # Multi-task Loss (NEW)
    'MultiTaskLoss',
    # Metrics
    'MultilabelAUC',
    'MultilabelMetrics',
    'compute_competition_metric',
    'LOCATION_NAMES',
    # Trainers
    'Trainer',
    'MultiTaskTrainer',  # NEW
]

