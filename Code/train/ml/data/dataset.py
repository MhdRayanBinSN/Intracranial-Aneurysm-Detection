"""
PyTorch Dataset classes for Intracranial Aneurysm Detection.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import yaml

from .preprocessing import DicomPreprocessor, get_modality_from_series
from .augmentations import get_train_transforms, get_val_transforms


# Target columns for prediction (14 total):
# - 13 anatomical locations (WHERE is the aneurysm?)
# - 1 summary column "Aneurysm Present" (DOES patient have aneurysm? = 1 if ANY location is 1)
LOCATION_COLUMNS = [
    # 13 Anatomical Locations
    'Left Infraclinoid Internal Carotid Artery',      # 1
    'Right Infraclinoid Internal Carotid Artery',     # 2
    'Left Supraclinoid Internal Carotid Artery',      # 3
    'Right Supraclinoid Internal Carotid Artery',     # 4
    'Left Middle Cerebral Artery',                    # 5
    'Right Middle Cerebral Artery',                   # 6
    'Anterior Communicating Artery',                  # 7
    'Left Anterior Cerebral Artery',                  # 8
    'Right Anterior Cerebral Artery',                 # 9
    'Left Posterior Communicating Artery',            # 10
    'Right Posterior Communicating Artery',           # 11
    'Basilar Tip',                                    # 12
    'Other Posterior Circulation',                    # 13
    # Summary Column (NOT a location)
    'Aneurysm Present',                               # 14 = OR of columns 1-13
]



class AneurysmDataset(Dataset):
    """
    PyTorch Dataset for Intracranial Aneurysm Detection.
    
    Loads DICOM series from a zip file and returns preprocessed volumes
    with multi-label targets.
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
        Initialize the dataset.
        
        Args:
            df: DataFrame with SeriesInstanceUID, Modality, and location columns
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
    
    def _get_zip_file(self) -> zipfile.ZipFile:
        """Lazy loading of zip file."""
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path, 'r')
        return self._zip_file
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            volume: Tensor of shape (1, D, H, W)
            labels: Tensor of shape (14,) with multi-hot encoding
        """
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        modality = row['Modality']
        
        # Expected output shape
        expected_shape = (self.preprocessor.num_slices, 
                         self.preprocessor.target_size[0], 
                         self.preprocessor.target_size[1])
        
        # Try to load from cache
        volume = self._load_from_cache(series_uid)
        
        # Verify cached volume has correct shape
        if volume is not None and volume.shape != expected_shape:
            volume = None  # Invalidate cache if wrong shape
        
        if volume is None:
            # Load and preprocess - use 'series/' folder structure
            series_path = f"series/{series_uid}"
            try:
                volume, _ = self.preprocessor.preprocess(
                    series_path, 
                    modality, 
                    self._get_zip_file()
                )
                # Verify output shape
                if volume.shape != expected_shape:
                    print(f"Warning: Volume {series_uid} has wrong shape {volume.shape}, expected {expected_shape}")
                    volume = np.zeros(expected_shape, dtype=np.float32)
            except Exception as e:
                # If loading fails, return a dummy volume (zeros)
                print(f"Warning: Failed to load series {series_uid}: {e}")
                volume = np.zeros(expected_shape, dtype=np.float32)
            
            # Save to cache only if successful
            if volume.shape == expected_shape:
                self._save_to_cache(series_uid, volume)
        
        # Apply transforms only if provided and volume is valid
        if self.transform is not None:
            try:
                volume = self.transform(volume)
            except Exception as e:
                print(f"Warning: Transform failed for {series_uid}: {e}")
                # Return normalized volume without augmentation
                volume = (volume - 0.5) / 0.5
        
        # Convert to tensor
        volume = torch.from_numpy(volume.copy()).float()
        
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        volume = volume.unsqueeze(0)
        
        # Get labels
        labels = row[LOCATION_COLUMNS].values.astype(np.float32)
        labels = torch.from_numpy(labels)
        
        return volume, labels
    
    def _load_from_cache(self, series_uid: str) -> Optional[np.ndarray]:
        """Load preprocessed volume from cache."""
        if self.cache_dir is None:
            return None
        
        cache_path = self.cache_dir / f"{series_uid}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def _save_to_cache(self, series_uid: str, volume: np.ndarray):
        """Save preprocessed volume to cache."""
        if self.cache_dir is None:
            return
        
        cache_path = self.cache_dir / f"{series_uid}.npy"
        np.save(cache_path, volume)
    
    def close(self):
        """Close the zip file."""
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None
    
    def __del__(self):
        self.close()


class AneurysmDataModule:
    """
    Data module for managing train/validation splits and data loaders.
    """
    
    def __init__(
        self,
        config_path: str,
    ):
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
        self.num_workers = self.config['data'].get('num_workers', 4)
        self.batch_size = self.config['training']['batch_size']
        
        # Initialize preprocessor
        self.preprocessor = DicomPreprocessor(
            target_size=tuple(self.config['data']['image_size']),
            num_slices=self.config['data']['num_slices'],
            ct_window_center=self.config['preprocessing']['ct_window_center'],
            ct_window_width=self.config['preprocessing']['ct_window_width'],
            mr_percentile_lower=self.config['preprocessing']['mr_percentile_lower'],
            mr_percentile_upper=self.config['preprocessing']['mr_percentile_upper'],
        )
        
        self.df_train = None
        self.df_val = None
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self):
        """Load data and create train/val splits."""
        # Load train.csv from zip
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
        
        # Stratified split based on Aneurysm Present
        from sklearn.model_selection import train_test_split
        
        self.df_train, self.df_val = train_test_split(
            df,
            train_size=self.train_split,
            stratify=df['Aneurysm Present'],
            random_state=42
        )
        
        print(f"Training samples: {len(self.df_train)}")
        print(f"Validation samples: {len(self.df_val)}")
        print(f"Positive cases (train): {self.df_train['Aneurysm Present'].sum()}")
        print(f"Positive cases (val): {self.df_val['Aneurysm Present'].sum()}")
        
        # Get transforms
        train_transforms = get_train_transforms(self.config) if self.config['augmentation']['enabled'] else None
        val_transforms = get_val_transforms(self.config)
        
        # Create datasets
        self.train_dataset = AneurysmDataset(
            df=self.df_train,
            zip_path=self.zip_path,
            preprocessor=self.preprocessor,
            transform=train_transforms,
            is_training=True,
            cache_dir=self.cache_dir,
        )
        
        self.val_dataset = AneurysmDataset(
            df=self.df_val,
            zip_path=self.zip_path,
            preprocessor=self.preprocessor,
            transform=val_transforms,
            is_training=False,
            cache_dir=self.cache_dir,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,  # Can use larger batch for val
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalance.
        
        Returns:
            Tensor of shape (14,) with class weights
        """
        if self.df_train is None:
            self.setup()
        
        pos_counts = self.df_train[LOCATION_COLUMNS].sum()
        neg_counts = len(self.df_train) - pos_counts
        
        # Weight = num_negative / num_positive
        weights = neg_counts / (pos_counts + 1)  # Add 1 to avoid division by zero
        weights = weights.clip(1, 100)  # Clip to reasonable range
        
        return torch.tensor(weights.values, dtype=torch.float32)


if __name__ == '__main__':
    # Example usage
    print("Dataset module loaded successfully!")
    print(f"Number of target columns: {len(LOCATION_COLUMNS)}")
    print(f"  - 13 anatomical locations (WHERE is the aneurysm?)")
    print(f"  - 1 summary column (Aneurysm Present = 1 if ANY location has aneurysm)")
    print("\nColumns:", LOCATION_COLUMNS)

