"""
Create nnU-Net Dataset Format (1st Place Solution - Phase 2)

Prepares data in nnU-Net v2 format for vessel segmentation training.
"""

import os
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


# nnU-Net environment variables
NNUNET_RAW = os.environ.get("nnUNet_raw", "./nnUNet_raw")
NNUNET_PREPROCESSED = os.environ.get("nnUNet_preprocessed", "./nnUNet_preprocessed")
NNUNET_RESULTS = os.environ.get("nnUNet_results", "./nnUNet_results")


class NNUNetDatasetCreator:
    """
    Create nnU-Net v2 format dataset for vessel segmentation.
    
    Dataset structure:
    nnUNet_raw/
    └── Dataset001_VesselSegmentation/
        ├── dataset.json
        ├── imagesTr/
        │   ├── case_000_0000.nii.gz
        │   └── ...
        ├── labelsTr/
        │   ├── case_000.nii.gz
        │   └── ...
        └── imagesTs/
            └── ...
    """
    
    def __init__(
        self,
        dataset_id: int = 1,
        dataset_name: str = "VesselSegmentation",
        raw_dir: str = NNUNET_RAW,
    ):
        """
        Initialize dataset creator.
        
        Args:
            dataset_id: nnU-Net dataset ID (e.g., 1, 3)
            dataset_name: Dataset name
            raw_dir: nnU-Net raw directory
        """
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.raw_dir = Path(raw_dir)
        
        # Dataset directory
        self.dataset_dir = self.raw_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
        self.images_tr = self.dataset_dir / "imagesTr"
        self.labels_tr = self.dataset_dir / "labelsTr"
        self.images_ts = self.dataset_dir / "imagesTs"
        
        # Create directories
        for d in [self.images_tr, self.labels_tr, self.images_ts]:
            d.mkdir(parents=True, exist_ok=True)
    
    def create_dataset_from_zip(
        self,
        zip_path: str,
        max_samples: Optional[int] = None,
        labels: Dict[int, str] = None,
    ):
        """
        Create nnU-Net dataset from competition zip file.
        
        Args:
            zip_path: Path to competition zip
            max_samples: Limit number of samples
            labels: Label mapping {id: name}
        """
        # Default vessel labels (from COW segmentation)
        if labels is None:
            labels = {
                0: "background",
                1: "Left ICA",
                2: "Right ICA",
                3: "Left MCA",
                4: "Right MCA",
                5: "AComm",
                6: "Left ACA",
                7: "Right ACA",
                8: "Left PComm",
                9: "Right PComm",
                10: "Left PCA",
                11: "Right PCA",
                12: "Basilar",
                13: "Other",
            }
        
        print(f"Creating nnU-Net dataset: {self.dataset_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Load train.csv
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
            
            # Find series with segmentation masks
            segmentation_files = [
                f for f in zf.namelist()
                if f.startswith('segmentations/') 
                and f.endswith('_cowseg.nii')
            ]
            
            # Get series UIDs with masks
            mask_uids = [
                Path(f).name.replace('_cowseg.nii', '')
                for f in segmentation_files
            ]
            
            # Filter to only series with masks
            df_with_masks = df[df['SeriesInstanceUID'].isin(mask_uids)]
            
            if max_samples:
                df_with_masks = df_with_masks.head(max_samples)
            
            print(f"Found {len(df_with_masks)} series with segmentation masks")
            
            # Process each series
            case_id = 0
            for _, row in tqdm(df_with_masks.iterrows(), total=len(df_with_masks)):
                series_uid = row['SeriesInstanceUID']
                
                try:
                    # Load and save image
                    image_saved = self._save_image_from_zip(
                        zf, series_uid, case_id
                    )
                    
                    # Load and save label
                    label_saved = self._save_label_from_zip(
                        zf, series_uid, case_id
                    )
                    
                    if image_saved and label_saved:
                        case_id += 1
                        
                except Exception as e:
                    print(f"Error processing {series_uid}: {e}")
            
            print(f"Created {case_id} training samples")
        
        # Create dataset.json
        self._create_dataset_json(case_id, labels)
    
    def _save_image_from_zip(
        self, zf: zipfile.ZipFile, series_uid: str, case_id: int
    ) -> bool:
        """Extract NIfTI image (the actual CT scan) from segmentations."""
        # The NIfTI files without _cowseg are the images
        nifti_path = f"segmentations/{series_uid}.nii"
        
        if nifti_path not in zf.namelist():
            return False
        
        # Output path (nnU-Net format: case_XXX_0000.nii.gz)
        output_path = self.images_tr / f"case_{case_id:03d}_0000.nii.gz"
        
        # Extract and save
        with zf.open(nifti_path) as src:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                tmp.write(src.read())
                tmp_path = tmp.name
        
        try:
            # Load, normalize, and save
            img = nib.load(tmp_path)
            volume = img.get_fdata().astype(np.float32)
            
            # Z-score normalization
            mean = volume.mean()
            std = volume.std()
            if std > 0:
                volume = (volume - mean) / std
            
            # Save as gzip compressed
            nib.save(
                nib.Nifti1Image(volume, img.affine, img.header),
                str(output_path)
            )
            return True
            
        finally:
            os.remove(tmp_path)
    
    def _save_label_from_zip(
        self, zf: zipfile.ZipFile, series_uid: str, case_id: int
    ) -> bool:
        """Extract segmentation mask from COW segmentation."""
        mask_path = f"segmentations/{series_uid}_cowseg.nii"
        
        if mask_path not in zf.namelist():
            return False
        
        # Output path (nnU-Net format: case_XXX.nii.gz)
        output_path = self.labels_tr / f"case_{case_id:03d}.nii.gz"
        
        # Extract and save
        with zf.open(mask_path) as src:
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                tmp.write(src.read())
                tmp_path = tmp.name
        
        try:
            img = nib.load(tmp_path)
            mask = img.get_fdata().astype(np.uint8)
            
            # Save as gzip compressed
            nib.save(
                nib.Nifti1Image(mask, img.affine, img.header),
                str(output_path)
            )
            return True
            
        finally:
            os.remove(tmp_path)
    
    def _create_dataset_json(self, num_training: int, labels: Dict[int, str]):
        """Create nnU-Net dataset.json file."""
        dataset_json = {
            "channel_names": {
                "0": "CT"
            },
            "labels": labels,
            "numTraining": num_training,
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "NibabelIOWithReorient"
        }
        
        json_path = self.dataset_dir / "dataset.json"
        with open(json_path, 'w') as f:
            json.dump(dataset_json, f, indent=4)
        
        print(f"Created {json_path}")


def create_splits_file(
    dataset_dir: str,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Create cross-validation splits file for nnU-Net.
    
    Args:
        dataset_dir: Path to dataset directory
        n_splits: Number of folds
        random_state: Random seed
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "imagesTr"
    
    # Get all case IDs
    cases = sorted([
        f.name.replace("_0000.nii.gz", "")
        for f in images_dir.glob("case_*_0000.nii.gz")
    ])
    
    # Create stratified splits (using dummy labels)
    labels = np.zeros(len(cases))  # All same class for now
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in kf.split(cases, labels):
        splits.append({
            "train": [cases[i] for i in train_idx],
            "val": [cases[i] for i in val_idx],
        })
    
    # Save splits
    splits_path = dataset_dir / "splits_final.json"
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Created {splits_path} with {n_splits} folds")


if __name__ == '__main__':
    print("=" * 60)
    print("NNUNET DATASET CREATION")
    print("=" * 60)
    
    ZIP_PATH = "C:/Users/Rayan/Desktop/Main Project/rsna-intracranial-aneurysm-detection.zip"
    
    # Create Dataset001 for vessel segmentation
    creator = NNUNetDatasetCreator(
        dataset_id=1,
        dataset_name="VesselSegmentation",
    )
    
    # Test with 10 samples
    print("\nCreating nnU-Net dataset (test with 10 samples)...")
    creator.create_dataset_from_zip(ZIP_PATH, max_samples=10)
    
    # Create splits
    create_splits_file(creator.dataset_dir, n_splits=5)
    
    print("\n✅ Dataset creation complete!")
    print(f"Dataset location: {creator.dataset_dir}")
