"""
Training module for Intracranial Aneurysm Detection.
Implements training loop with mixed precision, gradient accumulation, and checkpointing.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm
import numpy as np
import time
import yaml

from .losses import get_loss_function
from .metrics import MultilabelAUC, MultilabelMetrics


class Trainer:
    """
    Trainer class for model training with all bells and whistles.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader,
        val_loader,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 50)
        self.lr = train_config.get('learning_rate', 1e-4)
        self.weight_decay = train_config.get('weight_decay', 0.01)
        self.grad_accum_steps = train_config.get('gradient_accumulation', 4)
        self.mixed_precision = train_config.get('mixed_precision', True)
        self.early_stopping_patience = train_config.get('early_stopping_patience', 10)
        self.warmup_epochs = train_config.get('warmup_epochs', 5)
        
        # Checkpoint config
        checkpoint_config = config.get('checkpoint', {})
        self.save_dir = Path(checkpoint_config.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging config
        log_config = config.get('logging', {})
        self.log_dir = Path(log_config.get('log_dir', './logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_steps = log_config.get('log_every_n_steps', 10)
        
        # Initialize components
        self._setup_training()
    
    def _setup_training(self):
        """Initialize optimizer, scheduler, loss, and other components."""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        scheduler_type = self.config.get('training', {}).get('scheduler', 'cosine')
        total_steps = len(self.train_loader) * self.epochs // self.grad_accum_steps
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.lr * 0.01,
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=3,
            )
        elif scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.lr * 10,
                total_steps=total_steps,
            )
        else:
            self.scheduler = None
        
        # Loss function
        # Get class weights if configured
        pos_weight = None
        if self.config.get('training', {}).get('use_class_weights', False):
            # This would be computed from the data module
            pass
        
        self.criterion = get_loss_function(self.config, pos_weight)
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Metrics
        self.train_metrics = MultilabelAUC(num_classes=14)
        self.val_metrics = MultilabelAUC(num_classes=14)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Best metric tracking
        self.best_metric = 0.0
        self.epochs_without_improvement = 0
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, (volumes, labels) in enumerate(pbar):
            volumes = volumes.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(volumes)
                    loss = self.criterion(outputs['logits'], labels)
                    loss = loss / self.grad_accum_steps
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(volumes)
                loss = self.criterion(outputs['logits'], labels)
                loss = loss / self.grad_accum_steps
                loss.backward()
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler and isinstance(self.scheduler, (OneCycleLR, CosineAnnealingLR)):
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Log to tensorboard
                if self.global_step % self.log_every_n_steps == 0:
                    self.writer.add_scalar('train/loss', loss.item() * self.grad_accum_steps, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update metrics
            with torch.no_grad():
                self.train_metrics.update(labels, outputs['probabilities'])
            
            # Update progress bar
            pbar.set_postfix({'loss': total_loss / num_batches})
        
        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for volumes, labels in tqdm(self.val_loader, desc='Validating'):
            volumes = volumes.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(volumes)
            loss = self.criterion(outputs['logits'], labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            self.val_metrics.update(labels, outputs['probabilities'])
        
        # Compute metrics
        metrics = self.val_metrics.compute()
        metrics['loss'] = total_loss / num_batches
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            print(f"  New best model saved! (AUC: {self.best_metric:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Effective batch size: {len(self.train_loader.dataset) // len(self.train_loader) * self.grad_accum_steps}")
        print("-" * 50)
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log to tensorboard
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_auc', train_metrics['weighted_auc'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_auc', val_metrics['weighted_auc'], epoch)
            
            # Print metrics
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Train AUC: {train_metrics['weighted_auc']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val AUC: {val_metrics['weighted_auc']:.4f}")
            
            # Learning rate scheduler step (for plateau)
            if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['weighted_auc'])
            
            # Check for best model
            is_best = val_metrics['weighted_auc'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['weighted_auc']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        self.writer.close()
        print(f"\nTraining complete! Best validation AUC: {self.best_metric:.4f}")
        
        return self.best_metric


if __name__ == '__main__':
    print("Trainer module loaded successfully!")
