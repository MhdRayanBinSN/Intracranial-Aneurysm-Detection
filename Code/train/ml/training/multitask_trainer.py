"""
Multi-Task Trainer for Aneurysm Detection + Segmentation.

Trains a single model to do both:
1. Classification: Predict 14 location labels
2. Segmentation: Predict 3D aneurysm mask
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
import yaml
import time

from .losses import MultiTaskLoss, DiceLoss
from .metrics import MultilabelAUC


class MultiTaskTrainer:
    """
    Trainer for multi-task learning (Classification + Segmentation).
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Multi-task model (output: dict with 'classification' and 'segmentation')
            config: Configuration dictionary
            train_loader: Training data loader (returns: volume, mask, labels)
            val_loader: Validation data loader
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training settings
        self.epochs = config['training']['epochs']
        self.lr = config['training']['learning_rate']
        self.grad_accum = config['training'].get('gradient_accumulation', 1)
        self.mixed_precision = config['training'].get('mixed_precision', True)
        
        # Loss weights
        self.cls_weight = config['training'].get('cls_weight', 1.0)
        self.seg_weight = config['training'].get('seg_weight', 1.0)
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(
            cls_weight=self.cls_weight,
            seg_weight=self.seg_weight,
            cls_loss_type=config['training'].get('cls_loss', 'focal'),
            seg_loss_type=config['training'].get('seg_loss', 'dice_bce'),
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=config['training'].get('weight_decay', 0.01),
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision and device == 'cuda' else None
        
        # Metrics
        self.auc_metric = MultilabelAUC(num_classes=14)
        self.dice_metric = DiceLoss()  # Use Dice loss as metric (lower = better)
        
        # Logging
        log_dir = Path(config['logging']['log_dir']) / 'multitask'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = 0.0
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        warmup_epochs = self.config['training'].get('warmup_epochs', 5)
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs - warmup_epochs,
                eta_min=self.lr * 0.01,
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            return None
    
    def train(self) -> float:
        """
        Main training loop.
        
        Returns:
            Best validation AUC achieved
        """
        print("\n" + "=" * 60)
        print("MULTI-TASK TRAINING")
        print(f"  Classification weight: {self.cls_weight}")
        print(f"  Segmentation weight: {self.seg_weight}")
        print("=" * 60)
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_losses = self._train_epoch()
            
            # Validate
            val_metrics = self._validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log progress
            self._log_epoch(train_losses, val_metrics)
            
            # Save best model
            if val_metrics['auc'] > self.best_metric:
                self.best_metric = val_metrics['auc']
                self.save_checkpoint('best_multitask.pth')
                print(f"  ⭐ New best AUC: {self.best_metric:.4f}")
            
            # Early stopping check
            early_stop = self.config['training'].get('early_stopping_patience', 10)
            # ... (implement early stopping if needed)
        
        self.writer.close()
        return self.best_metric
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {'total': 0, 'classification': 0, 'segmentation': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (volumes, masks, labels) in enumerate(pbar):
            # Move to device
            volumes = volumes.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.mixed_precision and self.device == 'cuda':
                with autocast():
                    outputs = self.model(volumes)
                    losses = self.criterion(
                        outputs['classification'], labels,
                        outputs['segmentation'], masks,
                    )
                    loss = losses['total'] / self.grad_accum
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(volumes)
                losses = self.criterion(
                    outputs['classification'], labels,
                    outputs['segmentation'], masks,
                )
                loss = losses['total'] / self.grad_accum
                
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Accumulate losses
            total_losses['total'] += losses['total'].item()
            total_losses['classification'] += losses['classification'].item()
            total_losses['segmentation'] += losses['segmentation'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'cls': f"{losses['classification'].item():.4f}",
                'seg': f"{losses['segmentation'].item():.4f}",
            })
            
            self.global_step += 1
        
        # Average losses
        return {k: v / num_batches for k, v in total_losses.items()}
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        all_cls_preds = []
        all_cls_labels = []
        total_dice = 0
        num_batches = 0
        
        for volumes, masks, labels in tqdm(self.val_loader, desc="Validating"):
            volumes = volumes.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(volumes)
            
            # Classification predictions
            cls_probs = torch.sigmoid(outputs['classification'])
            all_cls_preds.append(cls_probs.cpu().numpy())
            all_cls_labels.append(labels.cpu().numpy())
            
            # Segmentation Dice score
            dice_loss = self.dice_metric(outputs['segmentation'], masks)
            total_dice += 1 - dice_loss.item()  # Convert loss to score
            num_batches += 1
        
        # Compute classification AUC
        all_preds = np.vstack(all_cls_preds)
        all_labels = np.vstack(all_cls_labels)
        
        # Update metric and compute
        self.auc_metric.reset()
        self.auc_metric.update(
            torch.from_numpy(all_labels),
            torch.from_numpy(all_preds)
        )
        auc_results = self.auc_metric.compute()
        auc = auc_results['weighted_auc']
        
        # Average Dice score
        avg_dice = total_dice / num_batches
        
        return {
            'auc': auc,
            'dice': avg_dice,
        }
    
    def _log_epoch(self, train_losses: Dict, val_metrics: Dict):
        """Log epoch results to tensorboard and console."""
        epoch = self.current_epoch + 1
        
        # Console output
        print(f"\nEpoch {epoch}/{self.epochs}")
        print(f"  Train Loss: {train_losses['total']:.4f} "
              f"(cls: {train_losses['classification']:.4f}, "
              f"seg: {train_losses['segmentation']:.4f})")
        print(f"  Val AUC: {val_metrics['auc']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        
        # Tensorboard
        self.writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
        self.writer.add_scalar('Loss/train_cls', train_losses['classification'], epoch)
        self.writer.add_scalar('Loss/train_seg', train_losses['segmentation'], epoch)
        self.writer.add_scalar('Metrics/val_auc', val_metrics['auc'], epoch)
        self.writer.add_scalar('Metrics/val_dice', val_metrics['dice'], epoch)
        self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == '__main__':
    print("MultiTaskTrainer module loaded successfully!")
    print("Use with: from training import MultiTaskTrainer")
