"""
Loss functions for multi-label classification and segmentation.

Classification losses:
- WeightedBCELoss: Binary cross entropy with class weights
- FocalLoss: Focuses on hard examples
- AsymmetricLoss: Different penalties for FP vs FN

Segmentation losses:
- DiceLoss: For segmentation task
- DiceBCELoss: Combined Dice + BCE

Multi-task losses:
- MultiTaskLoss: Combined classification + segmentation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss for handling class imbalance.
    
    Applies different weights to positive and negative samples per class.
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        """
        Args:
            pos_weight: Weight for positive samples per class (C,)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (B, C)
            targets: Binary targets (B, C)
        """
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Focuses on hard examples by down-weighting easy examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (B, C)
            targets: Binary targets (B, C)
        """
        probs = torch.sigmoid(logits)
        
        # Compute focal weight
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Apply focal weights
        loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Different penalties for false positives vs false negatives.
    Useful when missing a positive is worse than a false alarm.
    """
    
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = 'mean',
    ):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples (higher = more suppression)
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping threshold
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (B, C)
            targets: Binary targets (B, C)
        """
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping (for negative samples)
        probs_neg = (probs + self.clip).clamp(max=1)
        
        # Separate positive and negative losses
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        
        # Asymmetric focusing
        if self.gamma_pos > 0:
            loss_pos *= (1 - probs) ** self.gamma_pos
        if self.gamma_neg > 0:
            loss_neg *= probs_neg ** self.gamma_neg
        
        loss = -(loss_pos + loss_neg)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ============================================================
# SEGMENTATION LOSSES (NEW)
# ============================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Measures overlap between predicted and ground truth masks.
    Good for imbalanced segmentation (small objects).
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (B, 1, D, H, W)
            targets: Binary targets (B, 1, D, H, W)
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        # Compute intersection and union
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - Dice as loss (minimize loss = maximize Dice)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE Loss for segmentation.
    
    BCE helps with pixel-wise accuracy.
    Dice helps with overall shape overlap.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits (B, 1, D, H, W)
            targets: Binary targets (B, 1, D, H, W)
        """
        dice_loss = self.dice(logits, targets)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ============================================================
# MULTI-TASK LOSS (NEW)
# ============================================================

class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning (Classification + Segmentation).
    
    Total Loss = cls_weight * Classification Loss + seg_weight * Segmentation Loss
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        seg_weight: float = 1.0,
        cls_loss_type: str = 'focal',
        seg_loss_type: str = 'dice_bce',
    ):
        """
        Args:
            cls_weight: Weight for classification loss
            seg_weight: Weight for segmentation loss
            cls_loss_type: 'focal', 'bce', or 'asymmetric'
            seg_loss_type: 'dice', 'dice_bce', or 'bce'
        """
        super().__init__()
        
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        
        # Classification loss
        if cls_loss_type == 'focal':
            self.cls_loss = FocalLoss()
        elif cls_loss_type == 'bce':
            self.cls_loss = WeightedBCELoss()
        elif cls_loss_type == 'asymmetric':
            self.cls_loss = AsymmetricLoss()
        else:
            raise ValueError(f"Unknown cls_loss_type: {cls_loss_type}")
        
        # Segmentation loss
        if seg_loss_type == 'dice':
            self.seg_loss = DiceLoss()
        elif seg_loss_type == 'dice_bce':
            self.seg_loss = DiceBCELoss()
        elif seg_loss_type == 'bce':
            self.seg_loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown seg_loss_type: {seg_loss_type}")
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_logits: Classification logits (B, 14)
            cls_targets: Classification targets (B, 14)
            seg_logits: Segmentation logits (B, 1, D, H, W)
            seg_targets: Segmentation targets (B, 1, D, H, W)
            
        Returns:
            dict with 'total', 'classification', 'segmentation' losses
        """
        cls_loss = self.cls_loss(cls_logits, cls_targets)
        seg_loss = self.seg_loss(seg_logits, seg_targets)
        
        total_loss = self.cls_weight * cls_loss + self.seg_weight * seg_loss
        
        return {
            'total': total_loss,
            'classification': cls_loss,
            'segmentation': seg_loss,
        }


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions with weights.
    """
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        bce_weight: float = 0.0,
        asymmetric_weight: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.asymmetric_weight = asymmetric_weight
        
        if focal_weight > 0:
            self.focal = FocalLoss()
        if bce_weight > 0:
            self.bce = WeightedBCELoss(pos_weight=pos_weight)
        if asymmetric_weight > 0:
            self.asymmetric = AsymmetricLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss = 0
        
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        if self.asymmetric_weight > 0:
            loss = loss + self.asymmetric_weight * self.asymmetric(logits, targets)
        
        return loss


def get_loss_function(config: dict, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        config: Training configuration
        pos_weight: Class weights for imbalance
        
    Returns:
        Loss function
    """
    loss_type = config.get('training', {}).get('loss', 'focal')
    
    if loss_type == 'focal':
        gamma = config.get('training', {}).get('focal_gamma', 2.0)
        alpha = config.get('training', {}).get('focal_alpha', 0.25)
        return FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'bce':
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss()
    
    elif loss_type == 'combined':
        return CombinedLoss(
            focal_weight=1.0,
            bce_weight=0.5,
            pos_weight=pos_weight,
        )
    
    elif loss_type == 'multitask':
        return MultiTaskLoss(
            cls_weight=config.get('training', {}).get('cls_weight', 1.0),
            seg_weight=config.get('training', {}).get('seg_weight', 1.0),
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    print("=" * 60)
    print("LOSS FUNCTIONS TEST")
    print("=" * 60)
    
    # Create dummy predictions and targets
    cls_logits = torch.randn(4, 14)
    cls_targets = torch.randint(0, 2, (4, 14)).float()
    
    seg_logits = torch.randn(4, 1, 32, 128, 128)
    seg_targets = torch.randint(0, 2, (4, 1, 32, 128, 128)).float()
    
    # Test classification losses
    print("\n📊 Classification Losses:")
    cls_losses = {
        'BCE': WeightedBCELoss(),
        'Focal': FocalLoss(),
        'Asymmetric': AsymmetricLoss(),
    }
    
    for name, loss_fn in cls_losses.items():
        loss = loss_fn(cls_logits, cls_targets)
        print(f"  {name}: {loss.item():.4f}")
    
    # Test segmentation losses
    print("\n🎯 Segmentation Losses:")
    seg_losses = {
        'Dice': DiceLoss(),
        'Dice+BCE': DiceBCELoss(),
    }
    
    for name, loss_fn in seg_losses.items():
        loss = loss_fn(seg_logits, seg_targets)
        print(f"  {name}: {loss.item():.4f}")
    
    # Test multi-task loss
    print("\n🔧 Multi-Task Loss:")
    multitask_loss = MultiTaskLoss()
    losses = multitask_loss(cls_logits, cls_targets, seg_logits, seg_targets)
    print(f"  Classification: {losses['classification'].item():.4f}")
    print(f"  Segmentation: {losses['segmentation'].item():.4f}")
    print(f"  Total: {losses['total'].item():.4f}")
    
    print("\n✅ All loss functions working correctly!")

