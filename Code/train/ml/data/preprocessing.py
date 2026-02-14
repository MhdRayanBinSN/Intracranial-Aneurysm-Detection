"""
Data preprocessing utilities for DICOM medical imaging.
Handles loading, windowing, and normalization for CTA, MRA, and MRI modalities.
"""

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathlib import Path
from typing import Tuple, Optional, List, Union
import zipfile
import io
import os
import warnings

try:
    import SimpleITK as sitk
    HAS_SIMPLEITK = True
except ImportError:
    sitk = None
    HAS_SIMPLEITK = False
    warnings.warn("SimpleITK not available. Some preprocessing features may be limited.")


class DicomPreprocessor:
    """
    Preprocessor for DICOM medical images.
    Supports CTA, MRA, and MRI modalities with appropriate windowing.
    
    Handles known dataset issues:
    - Multi-frame DICOMs with missing attributes
    - Enhanced MR Image Storage format
    """
    
    # Default windowing parameters per modality
    WINDOW_PARAMS = {
        'CTA': {'center': 40, 'width': 400},      # Brain/vessel CT window
        'MRA': {'center': None, 'width': None},   # MRA uses percentile normalization
        'MRI': {'center': None, 'width': None},   # MRI uses percentile normalization
        'T1': {'center': None, 'width': None},
        'T2': {'center': None, 'width': None},
    }
    
    # Default parameters for multi-frame DICOMs with missing attributes
    # From Kaggle competition host recommendation
    DEFAULT_SPACING = (0.5, 0.5)
    DEFAULT_ORIGIN = (0, 0, 0)
    DEFAULT_ORIENTATION = [1, 0, 0, 0, 1, 0]
    DEFAULT_SLICE_THICKNESS = 5.0
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        num_slices: int = 64,
        ct_window_center: int = 40,
        ct_window_width: int = 400,
        mr_percentile_lower: float = 1,
        mr_percentile_upper: float = 99,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target (H, W) for each slice
            num_slices: Number of slices in output volume
            ct_window_center: Window center for CT images
            ct_window_width: Window width for CT images
            mr_percentile_lower: Lower percentile for MR normalization
            mr_percentile_upper: Upper percentile for MR normalization
        """
        self.target_size = target_size
        self.num_slices = num_slices
        self.ct_window_center = ct_window_center
        self.ct_window_width = ct_window_width
        self.mr_percentile_lower = mr_percentile_lower
        self.mr_percentile_upper = mr_percentile_upper
    
    def load_dicom_series(
        self,
        series_path: Union[str, Path],
        zip_file: Optional[zipfile.ZipFile] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Load a DICOM series from directory or zip file.
        
        Args:
            series_path: Path to series directory
            zip_file: Optional ZipFile object to read from
            
        Returns:
            volume: 3D numpy array (D, H, W)
            metadata: Dictionary with series metadata
        """
        if zip_file is not None:
            return self._load_from_zip(series_path, zip_file)
        else:
            return self._load_from_directory(series_path)
    
    def _load_from_directory(self, series_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
        """Load DICOM series from directory."""
        series_path = Path(series_path)
        dicom_files = sorted(series_path.glob('*.dcm'))
        
        if not dicom_files:
            # Try without extension
            dicom_files = [f for f in series_path.iterdir() if f.is_file()]
        
        slices = []
        metadata = {}
        
        for dcm_path in dicom_files:
            dcm = pydicom.dcmread(dcm_path)
            
            # Get metadata from first slice
            if not metadata:
                metadata = self._extract_metadata(dcm)
            
            # Get pixel array
            pixel_array = self._get_pixel_array(dcm)
            slices.append((dcm.InstanceNumber if hasattr(dcm, 'InstanceNumber') else 0, pixel_array))
        
        # Sort by instance number and stack
        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices], axis=0)
        
        return volume, metadata
    
    def _load_from_zip(
        self,
        series_path: str,
        zip_file: zipfile.ZipFile
    ) -> Tuple[np.ndarray, dict]:
        """Load DICOM series from zip file."""
        # List files in the series directory within the zip
        series_prefix = series_path.rstrip('/') + '/'
        dicom_files = [f for f in zip_file.namelist() if f.startswith(series_prefix) and not f.endswith('/')]
        
        slices = []
        metadata = {}
        
        for dcm_name in dicom_files:
            with zip_file.open(dcm_name) as f:
                dcm = pydicom.dcmread(io.BytesIO(f.read()))
                
                # Get metadata from first slice
                if not metadata:
                    metadata = self._extract_metadata(dcm)
                
                # Get pixel array
                pixel_array = self._get_pixel_array(dcm)
                instance_num = dcm.InstanceNumber if hasattr(dcm, 'InstanceNumber') else 0
                slices.append((instance_num, pixel_array))
        
        # Sort by instance number and stack
        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices], axis=0)
        
        return volume, metadata
    
    def _extract_metadata(self, dcm: pydicom.Dataset) -> dict:
        """
        Extract relevant metadata from DICOM.
        Uses default values for missing attributes (multi-frame DICOM issue fix).
        """
        # Check if this is a multi-frame Enhanced MR Image Storage
        is_multiframe = False
        sop_class = getattr(dcm, 'SOPClassUID', None)
        if sop_class and hasattr(sop_class, 'name'):
            is_multiframe = 'Enhanced' in sop_class.name
        
        # Get pixel spacing with default fallback
        pixel_spacing = getattr(dcm, 'PixelSpacing', None)
        if pixel_spacing is None:
            pixel_spacing = list(self.DEFAULT_SPACING)
        
        # Get slice thickness with default fallback
        slice_thickness = getattr(dcm, 'SliceThickness', None)
        if slice_thickness is None:
            slice_thickness = self.DEFAULT_SLICE_THICKNESS
        
        return {
            'PatientAge': getattr(dcm, 'PatientAge', 'Unknown'),
            'PatientSex': getattr(dcm, 'PatientSex', 'Unknown'),
            'Modality': getattr(dcm, 'Modality', 'Unknown'),
            'SeriesDescription': getattr(dcm, 'SeriesDescription', 'Unknown'),
            'Manufacturer': getattr(dcm, 'Manufacturer', 'Unknown'),
            'SliceThickness': slice_thickness,
            'PixelSpacing': pixel_spacing,
            'IsMultiframe': is_multiframe,
        }
    
    def _get_pixel_array(self, dcm: pydicom.Dataset) -> np.ndarray:
        """
        Extract pixel array from DICOM with proper VOI LUT applied.
        Handles multi-frame DICOMs and Enhanced MR Image Storage.
        """
        try:
            # Check for multi-frame DICOM
            if hasattr(dcm, 'NumberOfFrames') and int(dcm.NumberOfFrames) > 1:
                # Multi-frame DICOM - get all frames
                pixel_array = dcm.pixel_array
                if len(pixel_array.shape) == 3:
                    # Return first frame or handle appropriately
                    pixel_array = pixel_array[0] if pixel_array.shape[0] > 1 else pixel_array
            else:
                # Standard single-frame DICOM
                pixel_array = apply_voi_lut(dcm.pixel_array, dcm)
        except Exception:
            try:
                pixel_array = dcm.pixel_array
            except Exception:
                # If all else fails, return zeros
                rows = getattr(dcm, 'Rows', 512)
                cols = getattr(dcm, 'Columns', 512)
                pixel_array = np.zeros((rows, cols), dtype=np.float32)
        
        return pixel_array.astype(np.float32)
    
    def apply_windowing(
        self,
        volume: np.ndarray,
        modality: str
    ) -> np.ndarray:
        """
        Apply appropriate windowing based on modality.
        
        Args:
            volume: 3D numpy array (D, H, W)
            modality: One of 'CTA', 'MRA', 'MRI', 'T1', 'T2'
            
        Returns:
            Windowed volume normalized to [0, 1]
        """
        modality = modality.upper()
        
        if modality == 'CTA' or modality == 'CT':
            # Apply CT windowing
            return self._apply_ct_window(volume)
        else:
            # Apply percentile normalization for MR
            return self._apply_mr_normalization(volume)
    
    def _apply_ct_window(self, volume: np.ndarray) -> np.ndarray:
        """Apply CT windowing (for CTA)."""
        lower = self.ct_window_center - self.ct_window_width / 2
        upper = self.ct_window_center + self.ct_window_width / 2
        
        volume = np.clip(volume, lower, upper)
        volume = (volume - lower) / (upper - lower)
        
        return volume
    
    def _apply_mr_normalization(self, volume: np.ndarray) -> np.ndarray:
        """Apply percentile normalization (for MRA/MRI)."""
        lower = np.percentile(volume, self.mr_percentile_lower)
        upper = np.percentile(volume, self.mr_percentile_upper)
        
        volume = np.clip(volume, lower, upper)
        volume = (volume - lower) / (upper - lower + 1e-8)
        
        return volume
    
    def resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize volume to target dimensions.
        
        Args:
            volume: 3D numpy array (D, H, W)
            
        Returns:
            Resized volume (num_slices, target_size[0], target_size[1])
        """
        # Convert to SimpleITK for resampling
        sitk_volume = sitk.GetImageFromArray(volume)
        
        # Calculate new size
        original_size = sitk_volume.GetSize()
        original_spacing = sitk_volume.GetSpacing()
        
        new_size = [self.target_size[1], self.target_size[0], self.num_slices]
        new_spacing = [
            original_size[0] * original_spacing[0] / new_size[0],
            original_size[1] * original_spacing[1] / new_size[1],
            original_size[2] * original_spacing[2] / new_size[2],
        ]
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(sitk_volume.GetOrigin())
        resampler.SetOutputDirection(sitk_volume.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        
        resampled = resampler.Execute(sitk_volume)
        
        return sitk.GetArrayFromImage(resampled)
    
    def preprocess(
        self,
        series_path: Union[str, Path],
        modality: str,
        zip_file: Optional[zipfile.ZipFile] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Full preprocessing pipeline.
        
        Args:
            series_path: Path to DICOM series
            modality: Image modality
            zip_file: Optional ZipFile for reading from zip
            
        Returns:
            preprocessed_volume: Shape (num_slices, H, W), values in [0, 1]
            metadata: Series metadata dictionary
        """
        # Load DICOM series
        volume, metadata = self.load_dicom_series(series_path, zip_file)
        
        # Apply windowing
        volume = self.apply_windowing(volume, modality)
        
        # Resize to target dimensions
        volume = self.resize_volume(volume)
        
        return volume, metadata


def get_modality_from_series(series_description: str) -> str:
    """
    Infer modality from series description.
    
    Args:
        series_description: DICOM SeriesDescription
        
    Returns:
        Modality string
    """
    desc_lower = series_description.lower()
    
    if 'cta' in desc_lower or 'ct angio' in desc_lower:
        return 'CTA'
    elif 'mra' in desc_lower or 'mr angio' in desc_lower or 'tof' in desc_lower:
        return 'MRA'
    elif 't1' in desc_lower:
        return 'T1'
    elif 't2' in desc_lower:
        return 'T2'
    else:
        return 'MRI'


if __name__ == '__main__':
    # Example usage
    preprocessor = DicomPreprocessor(
        target_size=(256, 256),
        num_slices=64
    )
    
    print("Preprocessor initialized successfully!")
    print(f"Target size: {preprocessor.target_size}")
    print(f"Number of slices: {preprocessor.num_slices}")
