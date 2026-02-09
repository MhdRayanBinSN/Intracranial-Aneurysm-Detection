"""
Image Loader Utility
Loads medical images from various formats (DICOM, PNG, JPG, ZIP).
"""

import numpy as np
from PIL import Image
from pathlib import Path
import zipfile
import io
from typing import Union, List, Tuple, Optional

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


def load_dicom(path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """
    Load a DICOM file and return pixel array + metadata.
    
    Returns:
        pixel_array: HU values as float32
        metadata: dict with patient info
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required. Install with: pip install pydicom")
    
    dcm = pydicom.dcmread(str(path))
    
    # Get pixel array
    pixel_array = dcm.pixel_array.astype(np.float32)
    
    # Apply rescale to get Hounsfield Units
    slope = float(getattr(dcm, 'RescaleSlope', 1))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
    pixel_array = pixel_array * slope + intercept
    
    # Extract metadata
    metadata = {
        'PatientID': getattr(dcm, 'PatientID', 'Unknown'),
        'StudyDate': getattr(dcm, 'StudyDate', 'Unknown'),
        'Modality': getattr(dcm, 'Modality', 'Unknown'),
        'SeriesDescription': getattr(dcm, 'SeriesDescription', 'Unknown'),
        'SliceThickness': getattr(dcm, 'SliceThickness', None),
        'PixelSpacing': getattr(dcm, 'PixelSpacing', None),
    }
    
    return pixel_array, metadata


def load_dicom_from_zip(
    zip_path: Union[str, Path],
    series_uid: Optional[str] = None,
    slice_index: Optional[int] = None
) -> Tuple[np.ndarray, str]:
    """
    Load DICOM slice(s) from a ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        series_uid: Series UID to load (if None, picks first CTA)
        slice_index: Which slice to load (if None, loads middle slice)
    
    Returns:
        pixel_array: CT slice or volume
        series_uid: The series UID loaded
    """
    import pandas as pd
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find series
        if series_uid is None:
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
            series_uid = df[df['Modality'] == 'CTA']['SeriesInstanceUID'].values[0]
        
        # Get files
        series_path = f"series/{series_uid}/"
        dcm_files = sorted([
            f for f in zf.namelist()
            if f.startswith(series_path) and not f.endswith('/')
        ])
        
        if not dcm_files:
            raise ValueError(f"No DICOM files found for series {series_uid}")
        
        # Select slice
        if slice_index is None:
            slice_index = len(dcm_files) // 2
        
        dcm_file = dcm_files[min(slice_index, len(dcm_files) - 1)]
        
        # Load
        with zf.open(dcm_file) as f:
            dcm = pydicom.dcmread(io.BytesIO(f.read()))
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            pixel_array = pixel_array * slope + intercept
    
    return pixel_array, series_uid


def apply_ct_window(
    pixel_array: np.ndarray,
    center: int = 40,
    width: int = 400
) -> np.ndarray:
    """
    Apply CT windowing to convert HU to display values [0, 255].
    
    Args:
        pixel_array: CT image in Hounsfield Units
        center: Window center (default 40 for soft tissue)
        width: Window width (default 400)
    
    Returns:
        Windowed image as uint8 [0, 255]
    """
    lower = center - width / 2
    upper = center + width / 2
    
    windowed = np.clip(pixel_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    
    return windowed


def ct_to_pil(
    pixel_array: np.ndarray,
    window_center: int = 40,
    window_width: int = 400
) -> Image.Image:
    """
    Convert CT array to PIL Image for MedGemma input.
    
    Args:
        pixel_array: CT in HU
        window_center: CT window center
        window_width: CT window width
    
    Returns:
        RGB PIL Image
    """
    windowed = apply_ct_window(pixel_array, window_center, window_width)
    
    # Convert to RGB
    rgb = np.stack([windowed, windowed, windowed], axis=-1)
    
    return Image.fromarray(rgb)


if __name__ == "__main__":
    print("Image Loader Utilities")
    print("=" * 40)
    print(f"pydicom available: {PYDICOM_AVAILABLE}")
