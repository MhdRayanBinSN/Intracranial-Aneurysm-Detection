"""
Segmentation Dataset for Intracranial Aneurysm Segmentation.

This module handles loading NIfTI segmentation masks along with DICOM images.
Only 178 patients have segmentation masks available.

Author: Learning Project
Purpose: Research on Aneurysm Detection + Segmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import zipfile
import nibabel as nib
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict
import yaml

# Import from our preprocessing module
from .preprocessing import DicomPreprocessor


class SegmentationDataset(Dataset):
    """
    PyTorch Dataset for Aneurysm Segmentation Task.
    
    This dataset loads:
    1. DICOM images (input brain scans)
    2. NIfTI masks (ground truth segmentation)
    3. Classification labels (14 location labels)
    
    Only patients with segmentation masks are included (~178 patients).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        zip_path: str,
        preprocessor: Optional[DicomPreprocessor] = None,
        transform: Optional[Callable] = None,
        is_training: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the segmentation dataset.
        
        Args:
            df: DataFrame with SeriesInstanceUID, Modality, and location columns
                (filtered to only include patients with segmentation masks)
            zip_path: Path to the competition zip file
            preprocessor: DicomPreprocessor instance
            transform: Optional augmentation transforms
            is_training: Whether this is training data
            cache_dir: Optional directory to cache preprocessed volumes
        """
        self.df = df.reset_index(drop=True)
        self.zip_path = zip_path
        self.preprocessor = preprocessor or DicomPreprocessor()
        self.transform = transform
        self.is_training = is_training
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Open zip file
        self._zip_file = None
        
        # Get list of available segmentation masks
        self._available_masks = self._get_available_masks()
        
        # Filter df to only include patients with masks
        self.df = self.df[
            self.df['SeriesInstanceUID'].isin(self._available_masks)
        ].reset_index(drop=True)
        
        print(f"Segmentation dataset: {len(self.df)} patients with masks")
    
    def _get_zip_file(self) -> zipfile.ZipFile:
        """Lazy loading of zip file."""
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip_file
    
    def _get_available_masks(self) -> List[str]:
        """Get list of SeriesInstanceUIDs that have segmentation masks."""
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            nifti_files = [f for f in zf.namelist() 
                          if f.startswith('segmentations/') 
                          and f.endswith('.nii')
                          and '_cowseg' not in f]  # Exclude COW masks
            
            # Extract SeriesInstanceUID from filename
            mask_ids = [
                f.replace('segmentations/', '').replace('.nii', '') 
                for f in nifti_files
            ]
            
        return mask_ids
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            volume: Tensor of shape (1, D, H, W) - input image
            mask: Tensor of shape (1, D, H, W) - segmentation mask
            labels: Tensor of shape (14,) - classification labels
        """
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        modality = row['Modality']
        
        # Expected output shape
        expected_shape = (
            self.preprocessor.num_slices,
            self.preprocessor.target_size[0],
            self.preprocessor.target_size[1]
        )
        
        # ============================================================
        # STEP 1: Load DICOM volume (same as classification dataset)
        # ============================================================
        volume = self._load_volume(series_uid, modality, expected_shape)
        
        # ============================================================
        # STEP 2: Load NIfTI segmentation mask
        # ============================================================
        mask = self._load_mask(series_uid, expected_shape)
        
        # ============================================================
        # STEP 3: Apply transforms (same augmentation to both!)
        # ============================================================
        if self.transform is not None:
            # Important: Apply SAME random transform to both volume and mask
            seed = np.random.randint(0, 2**32)
            np.random.seed(seed)
            volume = self.transform(volume)
            np.random.seed(seed)  # Reset seed for same transform
            mask = self._transform_mask(mask)
        
        # ============================================================
        # STEP 4: Convert to tensors
        # ============================================================
        volume = torch.from_numpy(volume.copy()).float().unsqueeze(0)  # (1, D, H, W)
        mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)      # (1, D, H, W)
        
        # ============================================================
        # STEP 5: Get classification labels
        # ============================================================
        from .dataset import LOCATION_COLUMNS
        labels = row[LOCATION_COLUMNS].values.astype(np.float32)
        labels = torch.from_numpy(labels)  # (14,)
        
        return volume, mask, labels
    
    def _load_volume(
        self, 
        series_uid: str, 
        modality: str, 
        expected_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Load and preprocess DICOM volume."""
        # Try cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{series_uid}_vol.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load from zip
        series_path = f"series/{series_uid}"
        try:
            volume, _ = self.preprocessor.preprocess(
                series_path, modality, self._get_zip_file()
            )
        except Exception as e:
            print(f"Warning: Failed to load series {series_uid}: {e}")
            volume = np.zeros(expected_shape, dtype=np.float32)
        
        # Cache if directory specified
        if self.cache_dir:
            np.save(cache_path, volume)
        
        return volume
    
    def _load_mask(
        self, 
        series_uid: str, 
        expected_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Load NIfTI segmentation mask."""
        # Try cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"{series_uid}_mask.npy"
            if cache_path.exists():
                return np.load(cache_path)
        
        # Load from zip
        mask_path = f"segmentations/{series_uid}.nii"
        
        try:
            zf = self._get_zip_file()
            with zf.open(mask_path) as f:
                data = f.read()
                
                # NIfTI requires file on disk, use temp file
                with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                
                # Load with nibabel
                img = nib.load(tmp_path)
                mask = img.get_fdata()
                
                # Clean up temp file
                os.remove(tmp_path)
            
            # Convert to binary mask (aneurysm = 1, background = 0)
            # Note: The mask might contain CT values, need to threshold
            mask = (mask > 0).astype(np.float32)
            
            # Resize mask to match volume size
            mask = self._resize_mask(mask, expected_shape)
            
        except Exception as e:
            print(f"Warning: Failed to load mask {series_uid}: {e}")
            mask = np.zeros(expected_shape, dtype=np.float32)
        
        # Cache if directory specified
        if self.cache_dir:
            np.save(cache_path, mask)
        
        return mask
    
    def _resize_mask(
        self, 
        mask: np.ndarray, 
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Resize mask to target shape using nearest neighbor interpolation."""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1],
            target_shape[2] / mask.shape[2],
        )
        
        # Use order=0 (nearest neighbor) to preserve binary values
        resized = zoom(mask, zoom_factors, order=0)
        
        return resized.astype(np.float32)
    
    def _transform_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply geometric transforms to mask (no intensity transforms)."""
        # Only apply geometric transforms (flip, rotate)
        # Skip intensity transforms (noise, brightness)
        # This is handled by the seed synchronization in __getitem__
        return mask
    
    def close(self):
        """Close the zip file."""
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None
    
    def __del__(self):
        self.close()


class MultiTaskDataModule:
    """
    Data module for Multi-task learning (Classification + Segmentation).
    
    This module creates:
    1. Classification dataset: All 4348 patients
    2. Segmentation dataset: Only 178 patients with masks
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the data module.
        
        Args:
            config_path: Path to config.yaml
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.zip_path = self.config['data']['zip_path']
        self.cache_dir = self.config['data'].get('cache_dir')
        self.train_split = self.config['data'].get('train_split', 0.8)
        self.batch_size = self.config['training']['batch_size']
        
        # Initialize preprocessor
        self.preprocessor = DicomPreprocessor(
            target_size=tuple(self.config['data']['image_size']),
            num_slices=self.config['data']['num_slices'],
        )
        
        self.seg_train_dataset = None
        self.seg_val_dataset = None
    
    def setup_segmentation(self):
        """Setup segmentation datasets (only patients with masks)."""
        from sklearn.model_selection import train_test_split
        
        # Load train.csv
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
        
        # Get list of patients with segmentation masks
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            nifti_files = [f for f in zf.namelist() 
                          if f.startswith('segmentations/') 
                          and f.endswith('.nii')
                          and '_cowseg' not in f]
            mask_ids = [f.replace('segmentations/', '').replace('.nii', '') 
                       for f in nifti_files]
        
        # Filter to only patients with masks
        df_seg = df[df['SeriesInstanceUID'].isin(mask_ids)].copy()
        
        print(f"Patients with segmentation masks: {len(df_seg)}")
        
        # Split into train/val
        df_train, df_val = train_test_split(
            df_seg,
            train_size=self.train_split,
            stratify=df_seg['Aneurysm Present'],
            random_state=42
        )
        
        print(f"Segmentation train: {len(df_train)}")
        print(f"Segmentation val: {len(df_val)}")
        
        # Create datasets
        from .augmentations import get_train_transforms, get_val_transforms
        
        self.seg_train_dataset = SegmentationDataset(
            df=df_train,
            zip_path=self.zip_path,
            preprocessor=self.preprocessor,
            transform=get_train_transforms(self.config) if self.config['augmentation']['enabled'] else None,
            is_training=True,
            cache_dir=f"{self.cache_dir}_seg" if self.cache_dir else None,
        )
        
        self.seg_val_dataset = SegmentationDataset(
            df=df_val,
            zip_path=self.zip_path,
            preprocessor=self.preprocessor,
            transform=get_val_transforms(self.config),
            is_training=False,
            cache_dir=f"{self.cache_dir}_seg" if self.cache_dir else None,
        )
    
    def seg_train_dataloader(self) -> DataLoader:
        """Get segmentation training data loader."""
        return DataLoader(
            self.seg_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 0),
            pin_memory=True,
            drop_last=True,
        )
    
    def seg_val_dataloader(self) -> DataLoader:
        """Get segmentation validation data loader."""
        return DataLoader(
            self.seg_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 0),
            pin_memory=True,
        )


# ============================================================
# TEST SECTION - Run this file directly to test
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("SEGMENTATION DATASET TEST")
    print("=" * 60)
    
    # Test with default config
    config_path = "../config/config.yaml"
    
    from pathlib import Path
    if Path(config_path).exists():
        data_module = MultiTaskDataModule(config_path)
        data_module.setup_segmentation()
        
        # Get one sample
        sample = data_module.seg_train_dataset[0]
        volume, mask, labels = sample
        
        print(f"\nSample loaded successfully!")
        print(f"  Volume shape: {volume.shape}")  # (1, 32, 128, 128)
        print(f"  Mask shape: {mask.shape}")      # (1, 32, 128, 128)
        print(f"  Labels shape: {labels.shape}")  # (14,)
        print(f"  Mask unique values: {torch.unique(mask)}")
    else:
        print(f"Config not found: {config_path}")
        print("Run from ml/ folder: python -m data.segmentation_dataset")
