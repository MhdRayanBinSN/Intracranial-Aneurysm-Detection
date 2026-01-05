"""
Models module initialization.

Contains:
- Classification: AneurysmDetector (ResNet3D + classifier)
- Multi-task: MultiTaskUNet (classification + segmentation)
- ROI Classifier: 1st place solution architecture
"""

from .backbone import get_backbone, resnet3d_18, resnet3d_34, resnet3d_50, BACKBONES
from .classifier import MultiLabelClassifier, AnatomyAwareClassifier
from .model import AneurysmDetector, create_model, count_parameters
from .multitask_model import MultiTaskUNet, create_multitask_model
from .roi_classifier import (
    ROIClassifier,
    LocationAwareTransformer,
    VesselMaskedPoolingLayer,
    AuxiliarySphereLoss,
    create_roi_classifier,
)

__all__ = [
    # Backbones
    'get_backbone',
    'resnet3d_18',
    'resnet3d_34',
    'resnet3d_50',
    'BACKBONES',
    # Classifiers
    'MultiLabelClassifier',
    'AnatomyAwareClassifier',
    # Classification model
    'AneurysmDetector',
    'create_model',
    'count_parameters',
    # Multi-task model
    'MultiTaskUNet',
    'create_multitask_model',
    # ROI Classifier (1st place solution)
    'ROIClassifier',
    'LocationAwareTransformer',
    'VesselMaskedPoolingLayer',
    'AuxiliarySphereLoss',
    'create_roi_classifier',
]
