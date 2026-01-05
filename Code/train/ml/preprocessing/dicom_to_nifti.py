"""
DICOM to NIfTI Conversion (1st Place Solution - Phase 1)

Converts DICOM series to NIfTI format using dcm2niix.
Handles orientation standardization and z-score normalization.
"""

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm


class DicomToNiftiConverter:
    """
    Convert DICOM series to NIfTI format.
    Uses dcm2niix for conversion (same as 1st place solution).
    """
    
    def __init__(
        self,
        output_dir: str = "./data/series_nifti",
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize converter.
        
        Args:
            output_dir: Directory to save NIfTI files
            temp_dir: Temporary directory for extraction
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "dicom_temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_series_from_zip(
        self,
        zip_path: str,
        series_uid: str,
    ) -> Optional[Path]:
        """
        Convert a single DICOM series from zip to NIfTI.
        
        Args:
            zip_path: Path to competition zip file
            series_uid: Series Instance UID
            
        Returns:
            Path to NIfTI file if successful, None otherwise
        """
        # Output path
        output_file = self.output_dir / f"{series_uid}.nii.gz"
        
        # Skip if already exists
        if output_file.exists():
            return output_file
        
        # Extract DICOM files to temp directory
        series_temp = self.temp_dir / series_uid
        series_temp.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Find all DICOM files for this series
                series_prefix = f"series/{series_uid}/"
                dicom_files = [f for f in zf.namelist() 
                              if f.startswith(series_prefix) and not f.endswith('/')]
                
                if not dicom_files:
                    print(f"No DICOM files found for {series_uid}")
                    return None
                
                # Extract DICOM files
                for dcm_file in dicom_files:
                    filename = Path(dcm_file).name
                    with zf.open(dcm_file) as src:
                        with open(series_temp / filename, 'wb') as dst:
                            dst.write(src.read())
            
            # Run dcm2niix
            nifti_path = self._run_dcm2niix(series_temp, series_uid)
            
            if nifti_path and nifti_path.exists():
                # Move to output directory
                shutil.move(str(nifti_path), str(output_file))
                return output_file
            else:
                print(f"dcm2niix failed for {series_uid}")
                return None
                
        except Exception as e:
            print(f"Error converting {series_uid}: {e}")
            return None
        finally:
            # Cleanup temp files
            if series_temp.exists():
                shutil.rmtree(series_temp, ignore_errors=True)
    
    def _run_dcm2niix(self, input_dir: Path, series_uid: str) -> Optional[Path]:
        """
        Run dcm2niix on a directory of DICOM files.
        
        Args:
            input_dir: Directory containing DICOM files
            series_uid: Series UID for output filename
            
        Returns:
            Path to output NIfTI file
        """
        output_dir = self.temp_dir / "nifti_temp"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # dcm2niix command
        # -z y: compress (gzip)
        # -f: output filename
        # -o: output directory
        # -b n: no BIDS sidecar
        cmd = [
            "dcm2niix",
            "-z", "y",
            "-f", series_uid,
            "-o", str(output_dir),
            "-b", "n",
            str(input_dir)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Check for output file
            nifti_files = list(output_dir.glob(f"{series_uid}*.nii.gz"))
            
            if nifti_files:
                return nifti_files[0]
            
            # Try with Python dcm2niix if command line fails
            return self._run_dcm2niix_python(input_dir, series_uid, output_dir)
            
        except FileNotFoundError:
            # dcm2niix not in PATH, try Python version
            return self._run_dcm2niix_python(input_dir, series_uid, output_dir)
        except Exception as e:
            print(f"dcm2niix error: {e}")
            return None
    
    def _run_dcm2niix_python(
        self, input_dir: Path, series_uid: str, output_dir: Path
    ) -> Optional[Path]:
        """
        Fallback: Use Python dcm2niix package.
        """
        try:
            import dcm2niix
            
            dcm2niix.main([
                "-z", "y",
                "-f", series_uid,
                "-o", str(output_dir),
                "-b", "n",
                str(input_dir)
            ])
            
            nifti_files = list(output_dir.glob(f"{series_uid}*.nii.gz"))
            if nifti_files:
                return nifti_files[0]
            return None
            
        except Exception as e:
            print(f"Python dcm2niix error: {e}")
            return None
    
    def convert_all_from_zip(
        self,
        zip_path: str,
        csv_path: str = "train.csv",
        max_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Convert all DICOM series from zip to NIfTI.
        
        Args:
            zip_path: Path to competition zip
            csv_path: Path to train.csv within zip
            max_samples: Maximum number of samples to convert
            
        Returns:
            DataFrame with conversion results
        """
        # Load train.csv
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(csv_path) as f:
                df = pd.read_csv(f)
        
        if max_samples:
            df = df.head(max_samples)
        
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting DICOM to NIfTI"):
            series_uid = row['SeriesInstanceUID']
            
            nifti_path = self.convert_series_from_zip(zip_path, series_uid)
            
            results.append({
                'SeriesInstanceUID': series_uid,
                'nifti_path': str(nifti_path) if nifti_path else None,
                'success': nifti_path is not None,
            })
        
        results_df = pd.DataFrame(results)
        
        # Print summary
        success = results_df['success'].sum()
        total = len(results_df)
        print(f"\nConversion complete: {success}/{total} successful ({success/total*100:.1f}%)")
        
        return results_df


def normalize_volume_zscore(volume: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization (same as nnU-Net).
    
    Args:
        volume: 3D numpy array
        
    Returns:
        Normalized volume
    """
    mean = volume.mean()
    std = volume.std()
    
    if std > 0:
        return (volume - mean) / std
    else:
        return volume - mean


def load_and_normalize_nifti(nifti_path: str) -> Tuple[np.ndarray, nib.Nifti1Header]:
    """
    Load NIfTI file and apply z-score normalization.
    
    Args:
        nifti_path: Path to NIfTI file
        
    Returns:
        Normalized volume and header
    """
    img = nib.load(nifti_path)
    volume = img.get_fdata().astype(np.float32)
    volume = normalize_volume_zscore(volume)
    
    return volume, img.header


if __name__ == '__main__':
    print("=" * 60)
    print("DICOM TO NIFTI CONVERSION")
    print("=" * 60)
    
    # Configuration
    ZIP_PATH = "C:/Users/Rayan/Desktop/Main Project/rsna-intracranial-aneurysm-detection.zip"
    OUTPUT_DIR = "./data/series_nifti"
    
    # Create converter
    converter = DicomToNiftiConverter(output_dir=OUTPUT_DIR)
    
    # Test with 5 samples
    print("\nTesting with 5 samples...")
    results = converter.convert_all_from_zip(ZIP_PATH, max_samples=5)
    
    print("\nResults:")
    print(results)
