"""ROI extraction module for coarse-to-fine processing."""

from .roi_extraction import (
    ROIExtractor,
    VesselMaskedPooling,
)

__all__ = [
    'ROIExtractor',
    'VesselMaskedPooling',
]
