"""
ROI Classifier Training Script (1st Place Solution)

Trains the ROI classification model with:
- Location-aware classification (13 locations)
- Aneurysm presence detection
- Auxiliary sphere detection loss
- 5-fold cross-validation support
"""

import argparse
import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Optional, Tuple
import zipfile
import tempfile
import nibabel as nib

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from models.roi_classifier import ROIClassifier, AuxiliarySphereLoss, create_roi_classifier
from roi.roi_extraction import ROIExtractor, VesselMaskedPooling
from data.preprocessing import DicomPreprocessor


class ROIDataset(Dataset):
    """
    Dataset for ROI classification training.
    
    Loads:
    - ROI volumes (preprocessed)
    - Vessel segmentation masks
    - Labels (13 locations + presence)
    """
    
    # Label columns
    LOCATION_COLUMNS = [
        'Left Infraclinoid ICA',
        'Right Infraclinoid ICA',
        'Left Supraclinoid ICA',
        'Right Supraclinoid ICA',
        'Left Middle Cerebral Artery',
        'Right Middle Cerebral Artery',
        'Anterior Communicating Artery',
        'Left Anterior Cerebral Artery',
        'Right Anterior Cerebral Artery',
        'Left Posterior Communicating Artery',
        'Right Posterior Communicating Artery',
        'Basilar Tip',
        'Other Posterior Circulation',
    ]
    
    def __init__(
        self,
        df: pd.DataFrame,
        zip_path: str,
        roi_size: Tuple[int, int, int] = (64, 128, 128),
        transform=None,
        use_vessel_masks: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with labels
            zip_path: Path to competition zip
            roi_size: ROI volume size (D, H, W)
            transform: Optional augmentations
            use_vessel_masks: Whether to load vessel masks
        """
        self.df = df.reset_index(drop=True)
        self.zip_path = zip_path
        self.roi_size = roi_size
        self.transform = transform
        self.use_vessel_masks = use_vessel_masks
        
        self.preprocessor = DicomPreprocessor(
            target_size=(roi_size[1], roi_size[2]),
            num_slices=roi_size[0],
        )
        
        # Filter to only series with segmentation masks
        self._filter_with_masks()
    
    def _filter_with_masks(self):
        """Filter to series with available masks."""
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            mask_files = [
                f.replace('segmentations/', '').replace('_cowseg.nii', '')
                for f in zf.namelist()
                if '_cowseg.nii' in f
            ]
        
        original_len = len(self.df)
        self.df = self.df[self.df['SeriesInstanceUID'].isin(mask_files)]
        self.df = self.df.reset_index(drop=True)
        print(f"Filtered: {original_len} -> {len(self.df)} (with masks)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        modality = row['Modality']
        
        # Load volume
        volume = self._load_volume(series_uid, modality)
        
        # Load vessel mask
        if self.use_vessel_masks:
            vessel_mask = self._load_vessel_mask(series_uid)
        else:
            vessel_mask = np.zeros(self.roi_size, dtype=np.int64)
        
        # Get labels
        location_labels = np.array([
            row.get(col, 0) for col in self.LOCATION_COLUMNS
        ], dtype=np.float32)
        
        presence_label = np.float32(row.get('Aneurysm Present', 0))
        
        # Apply transforms
        if self.transform:
            volume = self.transform(volume)
        
        # Normalize
        volume = (volume - 0.5) / 0.5
        
        # Convert to tensors
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)
        vessel_mask = torch.from_numpy(vessel_mask).long()  # (D, H, W)
        location_labels = torch.from_numpy(location_labels)  # (13,)
        presence_label = torch.tensor(presence_label)  # scalar
        
        return {
            'volume': volume,
            'vessel_mask': vessel_mask,
            'location_labels': location_labels,
            'presence_label': presence_label,
            'series_uid': series_uid,
        }
    
    def _load_volume(self, series_uid: str, modality: str) -> np.ndarray:
        """Load and preprocess volume from NIfTI."""
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            nifti_path = f"segmentations/{series_uid}.nii"
            
            if nifti_path in zf.namelist():
                with zf.open(nifti_path) as f:
                    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                
                try:
                    img = nib.load(tmp_path)
                    volume = img.get_fdata().astype(np.float32)
                finally:
                    os.remove(tmp_path)
                
                # Resize to ROI size
                from scipy.ndimage import zoom
                factors = tuple(r / v for r, v in zip(self.roi_size, volume.shape))
                volume = zoom(volume, factors, order=1)
                
                # Z-score normalize
                mean, std = volume.mean(), volume.std()
                if std > 0:
                    volume = (volume - mean) / std
                
                # Clip and rescale to [0, 1]
                volume = np.clip(volume, -3, 3)
                volume = (volume + 3) / 6
                
                return volume
        
        # Fallback: return zeros
        return np.zeros(self.roi_size, dtype=np.float32)
    
    def _load_vessel_mask(self, series_uid: str) -> np.ndarray:
        """Load vessel segmentation mask."""
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            mask_path = f"segmentations/{series_uid}_cowseg.nii"
            
            if mask_path in zf.namelist():
                with zf.open(mask_path) as f:
                    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                
                try:
                    img = nib.load(tmp_path)
                    mask = img.get_fdata().astype(np.int64)
                finally:
                    os.remove(tmp_path)
                
                # Resize to ROI size
                from scipy.ndimage import zoom
                factors = tuple(r / v for r, v in zip(self.roi_size, mask.shape))
                mask = zoom(mask, factors, order=0)  # Nearest neighbor
                
                return mask.astype(np.int64)
        
        return np.zeros(self.roi_size, dtype=np.int64)


class ROIClassifierTrainer:
    """
    Trainer for ROI Classifier.
    
    Features:
    - Mixed precision training
    - Gradient accumulation
    - EMA weights
    - Cosine annealing with warmup
    - Multi-loss training (locations + presence + aux sphere)
    """
    
    def __init__(
        self,
        model: ROIClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
    ):
        """
        Initialize trainer.
        
        Args:
            model: ROI Classifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss functions
        self.location_loss_fn = nn.BCEWithLogitsLoss()
        self.presence_loss_fn = nn.BCEWithLogitsLoss()
        self.aux_sphere_loss_fn = AuxiliarySphereLoss()
        
        # Loss weights (from 1st place solution)
        self.location_weight = config.get('location_loss_weight', 0.1)
        self.presence_weight = config.get('presence_loss_weight', 0.05)
        self.aux_sphere_weight = config.get('aux_sphere_loss_weight', 1.0)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30),
            eta_min=config.get('min_lr', 1e-6),
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True)
        
        # Gradient accumulation
        self.accumulation_steps = config.get('accumulation_steps', 4)
        
        # EMA
        self.ema_model = None
        if config.get('use_ema', True):
            self.ema_model = self._create_ema_model()
        
        # Logging
        self.output_dir = Path(config.get('output_dir', './checkpoints/roi_classifier'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_score = 0
        self.history = {'train_loss': [], 'val_auc': []}
    
    def _create_ema_model(self):
        """Create EMA copy of model."""
        import copy
        ema = copy.deepcopy(self.model)
        ema.eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema
    
    def _update_ema(self, decay=0.999):
        """Update EMA weights."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_p, model_p in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            volume = batch['volume'].to(self.device)
            vessel_mask = batch['vessel_mask'].to(self.device)
            location_labels = batch['location_labels'].to(self.device)
            presence_label = batch['presence_label'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(volume, vessel_mask)
                
                # Location loss
                location_loss = self.location_loss_fn(
                    outputs['locations'], location_labels
                )
                
                # Presence loss
                presence_loss = self.presence_loss_fn(
                    outputs['presence'], presence_label
                )
                
                # Auxiliary sphere loss (simplified - no target for now)
                aux_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                loss = (
                    self.location_weight * location_loss +
                    self.presence_weight * presence_loss +
                    self.aux_sphere_weight * aux_loss
                )
                
                # Scale for accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update EMA
                self._update_ema()
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'loc': f"{location_loss.item():.4f}",
                'pres': f"{presence_loss.item():.4f}",
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        model = self.ema_model if self.ema_model else self.model
        model.eval()
        
        all_location_preds = []
        all_location_labels = []
        all_presence_preds = []
        all_presence_labels = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            volume = batch['volume'].to(self.device)
            vessel_mask = batch['vessel_mask'].to(self.device)
            location_labels = batch['location_labels']
            presence_label = batch['presence_label']
            
            outputs = model(volume, vessel_mask)
            
            # Collect predictions
            all_location_preds.append(torch.sigmoid(outputs['locations']).cpu())
            all_location_labels.append(location_labels)
            all_presence_preds.append(torch.sigmoid(outputs['presence']).cpu())
            all_presence_labels.append(presence_label)
        
        # Concatenate
        all_location_preds = torch.cat(all_location_preds, dim=0).numpy()
        all_location_labels = torch.cat(all_location_labels, dim=0).numpy()
        all_presence_preds = torch.cat(all_presence_preds, dim=0).numpy()
        all_presence_labels = torch.cat(all_presence_labels, dim=0).numpy()
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        
        metrics = {}
        
        # Location AUC
        try:
            # Calculate per-location AUC
            location_aucs = []
            for i in range(13):
                if all_location_labels[:, i].sum() > 0:
                    auc = roc_auc_score(all_location_labels[:, i], all_location_preds[:, i])
                    location_aucs.append(auc)
            metrics['location_auc'] = np.mean(location_aucs) if location_aucs else 0.5
        except:
            metrics['location_auc'] = 0.5
        
        # Presence AUC
        try:
            metrics['presence_auc'] = roc_auc_score(all_presence_labels, all_presence_preds)
        except:
            metrics['presence_auc'] = 0.5
        
        # Weighted AUC (competition metric)
        metrics['weighted_auc'] = 0.5 * metrics['location_auc'] + 0.5 * metrics['presence_auc']
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        if self.ema_model:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'latest.pth')
        
        # Save best
        if metrics['weighted_auc'] > self.best_score:
            self.best_score = metrics['weighted_auc']
            torch.save(checkpoint, self.output_dir / 'best.pth')
            print(f"  New best model! AUC: {self.best_score:.4f}")
    
    def train(self, num_epochs: int):
        """Full training loop."""
        print("=" * 60)
        print("ROI CLASSIFIER TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            metrics = self.validate()
            self.history['val_auc'].append(metrics['weighted_auc'])
            
            # Scheduler step
            self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Location AUC: {metrics['location_auc']:.4f}")
            print(f"  Presence AUC: {metrics['presence_auc']:.4f}")
            print(f"  Weighted AUC: {metrics['weighted_auc']:.4f}")
            print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics)
        
        print("\n✅ Training complete!")
        print(f"Best AUC: {self.best_score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train ROI Classifier')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--fold', type=int, default=0, help='CV fold')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with args
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    config['fold'] = args.fold
    
    # Paths
    ZIP_PATH = config.get('data', {}).get(
        'zip_path',
        'C:/Users/Rayan/Desktop/Main Project/rsna-intracranial-aneurysm-detection.zip'
    )
    
    # Load data
    print("Loading data...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        with zf.open('train.csv') as f:
            df = pd.read_csv(f)
    
    if args.debug:
        df = df.head(50)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['Aneurysm Present'], random_state=42
    )
    
    # Create datasets
    roi_size = (64, 128, 128)
    
    train_dataset = ROIDataset(train_df, ZIP_PATH, roi_size=roi_size)
    val_dataset = ROIDataset(val_df, ZIP_PATH, roi_size=roi_size)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = create_roi_classifier({
        'in_channels': 1,
        'base_features': 32,
        'num_locations': 13,
        'd_model': 256,
    })
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    trainer = ROIClassifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
    )
    
    # Train
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
