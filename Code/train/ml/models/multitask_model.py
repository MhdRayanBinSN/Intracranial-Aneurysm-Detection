"""
Multi-Task Model for Aneurysm Detection + Segmentation.

This model does TWO tasks simultaneously:
1. Classification: Predicts 14 labels (13 locations + 1 presence)
2. Segmentation: Predicts 3D mask of aneurysm

Architecture:
- Shared Encoder (3D ResNet backbone)
- Classification Head (Global pooling + FC layers)
- Segmentation Decoder (U-Net style upsampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class ConvBlock3D(nn.Module):
    """3D Convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Encoder block: Conv + MaxPool."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features  # Return both for skip connections


class UpBlock(nn.Module):
    """Decoder block: Upsample + Concat + Conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_channels * 2, out_channels)  # *2 for skip connection
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch (if any)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)  # Concat on channel dimension
        return self.conv(x)


class MultiTaskUNet(nn.Module):
    """
    Multi-Task 3D U-Net for Classification + Segmentation.
    
    Input: (B, 1, D, H, W) - 3D brain volume
    Output:
        - classification: (B, 14) - location probabilities
        - segmentation: (B, 1, D, H, W) - aneurysm mask
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 14,
        base_features: int = 32,
        dropout: float = 0.3,
    ):
        """
        Initialize the multi-task model.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            num_classes: Number of classification outputs (14)
            base_features: Base number of features (doubles each level)
            dropout: Dropout rate for classification head
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Feature sizes at each level
        f = base_features  # 32
        
        # ============================================================
        # ENCODER (Shared between classification and segmentation)
        # ============================================================
        self.enc1 = DownBlock(in_channels, f)        # 1 -> 32
        self.enc2 = DownBlock(f, f * 2)              # 32 -> 64
        self.enc3 = DownBlock(f * 2, f * 4)          # 64 -> 128
        self.enc4 = DownBlock(f * 4, f * 8)          # 128 -> 256
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(f * 8, f * 16)  # 256 -> 512
        
        # ============================================================
        # CLASSIFICATION HEAD
        # ============================================================
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f * 16, f * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(f * 4, num_classes),
        )
        
        # ============================================================
        # SEGMENTATION DECODER
        # ============================================================
        self.dec4 = UpBlock(f * 16, f * 8)    # 512 -> 256
        self.dec3 = UpBlock(f * 8, f * 4)     # 256 -> 128
        self.dec2 = UpBlock(f * 4, f * 2)     # 128 -> 64
        self.dec1 = UpBlock(f * 2, f)         # 64 -> 32
        
        # Final segmentation output
        self.seg_out = nn.Conv3d(f, 1, kernel_size=1)  # 32 -> 1
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, D, H, W)
            
        Returns:
            dict with:
                'classification': (B, 14) classification logits
                'segmentation': (B, 1, D, H, W) segmentation logits
        """
        # ============================================================
        # ENCODER
        # ============================================================
        x, skip1 = self.enc1(x)    # skip1: (B, 32, D, H, W)
        x, skip2 = self.enc2(x)    # skip2: (B, 64, D/2, H/2, W/2)
        x, skip3 = self.enc3(x)    # skip3: (B, 128, D/4, H/4, W/4)
        x, skip4 = self.enc4(x)    # skip4: (B, 256, D/8, H/8, W/8)
        
        # Bottleneck
        x = self.bottleneck(x)     # (B, 512, D/16, H/16, W/16)
        
        # ============================================================
        # CLASSIFICATION HEAD
        # ============================================================
        cls_features = self.global_pool(x)  # (B, 512, 1, 1, 1)
        classification = self.classifier(cls_features)  # (B, 14)
        
        # ============================================================
        # SEGMENTATION DECODER
        # ============================================================
        x = self.dec4(x, skip4)    # (B, 256, D/8, H/8, W/8)
        x = self.dec3(x, skip3)    # (B, 128, D/4, H/4, W/4)
        x = self.dec2(x, skip2)    # (B, 64, D/2, H/2, W/2)
        x = self.dec1(x, skip1)    # (B, 32, D, H, W)
        
        segmentation = self.seg_out(x)  # (B, 1, D, H, W)
        
        return {
            'classification': classification,
            'segmentation': segmentation,
        }


def create_multitask_model(config: dict) -> MultiTaskUNet:
    """
    Create a multi-task model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MultiTaskUNet model
    """
    return MultiTaskUNet(
        in_channels=config['model'].get('in_channels', 1),
        num_classes=config['model'].get('num_classes', 14),
        base_features=config['model'].get('base_features', 32),
        dropout=config['model'].get('dropout', 0.3),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# TEST
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MULTI-TASK MODEL TEST")
    print("=" * 60)
    
    # Create model
    model = MultiTaskUNet(
        in_channels=1,
        num_classes=14,
        base_features=32,
    )
    
    print(f"Model created successfully!")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 32, 128, 128)  # Batch of 2
    print(f"\nInput shape: {x.shape}")
    
    output = model(x)
    print(f"Classification output: {output['classification'].shape}")  # (2, 14)
    print(f"Segmentation output: {output['segmentation'].shape}")      # (2, 1, 32, 128, 128)
    
    print("\n✅ Multi-task model working correctly!")
