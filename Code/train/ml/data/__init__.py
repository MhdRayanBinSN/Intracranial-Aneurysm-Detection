"""
Data module initialization.

Contains:
- Classification: AneurysmDataset, AneurysmDataModule
- Segmentation: SegmentationDataset, MultiTaskDataModule
"""

from .preprocessing import DicomPreprocessor, get_modality_from_series
from .dataset import AneurysmDataset, AneurysmDataModule, LOCATION_COLUMNS
from .segmentation_dataset import SegmentationDataset, MultiTaskDataModule
from .augmentations import get_train_transforms, get_val_transforms

__all__ = [
    # Preprocessing
    'DicomPreprocessor',
    'get_modality_from_series',
    # Classification
    'AneurysmDataset',
    'AneurysmDataModule',
    'LOCATION_COLUMNS',
    # Segmentation (NEW)
    'SegmentationDataset',
    'MultiTaskDataModule',
    # Augmentations
    'get_train_transforms',
    'get_val_transforms',
]

