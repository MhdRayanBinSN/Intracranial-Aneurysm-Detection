"""
ROI Classifier with Location-Aware Transformer (1st Place Solution - Phase 4)

Key components:
1. nnU-Net pretrained encoder
2. Simplified decoder
3. Vessel Region-Masked Pooling
4. Location-Aware Transformer
5. Auxiliary sphere detection head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class LocationAwareTransformer(nn.Module):
    """
    Transformer for modeling inter-location relationships.
    
    Takes features from 13 anatomical locations and models
    their relationships using self-attention.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize Location-Aware Transformer.
        
        Args:
            d_model: Feature dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Hidden dimension in FFN
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Learnable position embeddings for 13 locations
        self.pos_embedding = nn.Parameter(torch.randn(1, 13, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Location features (B, 13, d_model)
            
        Returns:
            Enhanced features (B, 13, d_model)
        """
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Transformer
        x = self.transformer(x)
        
        # Normalize
        x = self.norm(x)
        
        return x


class VesselMaskedPoolingLayer(nn.Module):
    """
    PyTorch layer for vessel-masked feature pooling.
    
    Extracts features for each anatomical location using vessel masks.
    """
    
    def __init__(self, num_locations: int = 13):
        super().__init__()
        self.num_locations = num_locations
    
    def forward(
        self,
        features: torch.Tensor,
        vessel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool features for each location.
        
        Args:
            features: Feature map (B, C, D, H, W)
            vessel_mask: Vessel mask with location labels (B, D', H', W')
            
        Returns:
            Location features (B, 13, C)
        """
        B, C, D, H, W = features.shape
        device = features.device
        
        # Resize mask to match features if needed
        if vessel_mask.shape[1:] != (D, H, W):
            vessel_mask = F.interpolate(
                vessel_mask.unsqueeze(1).float(),
                size=(D, H, W),
                mode='nearest'
            ).squeeze(1).long()
        
        # Pool for each location
        location_features = torch.zeros(B, self.num_locations, C, device=device)
        
        for loc in range(1, self.num_locations + 1):
            loc_mask = (vessel_mask == loc)  # (B, D, H, W)
            
            for b in range(B):
                if loc_mask[b].sum() > 0:
                    # Masked mean pooling
                    mask_expanded = loc_mask[b].unsqueeze(0).expand(C, -1, -1, -1)
                    masked_features = features[b] * mask_expanded.float()
                    location_features[b, loc - 1] = masked_features.sum(dim=(1, 2, 3)) / loc_mask[b].sum()
        
        return location_features


class AuxiliarySphereLoss(nn.Module):
    """
    Auxiliary loss for aneurysm sphere detection.
    
    Creates a small sphere at aneurysm location as training target.
    Uses Balanced BCE + Focal-Tversky++ loss.
    """
    
    def __init__(
        self,
        sphere_radius: int = 5,
        alpha: float = 0.7,
        gamma: float = 0.75,
    ):
        """
        Initialize loss.
        
        Args:
            sphere_radius: Radius of target sphere in voxels
            alpha: Tversky alpha (recall emphasis)
            gamma: Focal-Tversky gamma
        """
        super().__init__()
        self.sphere_radius = sphere_radius
        self.alpha = alpha
        self.gamma = gamma
    
    def create_sphere_target(
        self,
        shape: Tuple[int, int, int],
        center: Tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Create sphere target centered at given location."""
        d, h, w = shape
        
        # Create coordinate grids
        z = torch.arange(d, device=device).view(-1, 1, 1)
        y = torch.arange(h, device=device).view(1, -1, 1)
        x = torch.arange(w, device=device).view(1, 1, -1)
        
        # Distance from center
        dist = ((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2).sqrt()
        
        # Binary sphere
        sphere = (dist <= self.sphere_radius).float()
        
        return sphere
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Balanced BCE + Focal-Tversky++ loss.
        
        Args:
            pred: Predicted sphere logits (B, 1, D, H, W)
            target: Target sphere mask (B, 1, D, H, W)
        """
        # Sigmoid activation
        pred_prob = torch.sigmoid(pred)
        
        # Balanced BCE
        pos_weight = (target == 0).sum() / (target == 1).sum().clamp(min=1)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )
        
        # Focal-Tversky++
        tversky_loss = self.focal_tversky_plus(pred_prob, target)
        
        return bce_loss + tversky_loss
    
    def focal_tversky_plus(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal-Tversky++ loss for sparse targets."""
        smooth = 1e-6
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Tversky index
        TP = (pred_flat * target_flat).sum()
        FP = ((1 - target_flat) * pred_flat).sum()
        FN = (target_flat * (1 - pred_flat)).sum()
        
        tversky = (TP + smooth) / (TP + self.alpha * FN + (1 - self.alpha) * FP + smooth)
        
        # Focal modifier
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class ROIClassifier(nn.Module):
    """
    ROI Classification model (1st place solution architecture).
    
    Components:
    - Encoder: 3D CNN (can be pretrained nnU-Net)
    - Decoder: Simplified for efficiency
    - Vessel-Masked Pooling: Extract location features
    - Location-Aware Transformer: Model relationships
    - Classification heads: 13 locations + Aneurysm Present
    - Auxiliary head: Sphere detection
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 32,
        num_locations: int = 13,
        d_model: int = 256,
        dropout: float = 0.3,
    ):
        """
        Initialize ROI Classifier.
        
        Args:
            in_channels: Number of input channels
            base_features: Base feature count
            num_locations: Number of anatomical locations
            d_model: Transformer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_locations = num_locations
        self.d_model = d_model
        
        # Encoder (similar to nnU-Net encoder)
        self.encoder = nn.ModuleList([
            self._conv_block(in_channels, base_features),
            self._conv_block(base_features, base_features * 2),
            self._conv_block(base_features * 2, base_features * 4),
            self._conv_block(base_features * 4, base_features * 8),
            self._conv_block(base_features * 8, base_features * 16),
        ])
        
        self.pool = nn.MaxPool3d(2)
        
        # Simplified decoder (removed final block for efficiency)
        decoder_features = base_features * 8
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_features * 16, decoder_features, 2, 2),
            self._conv_block(decoder_features, decoder_features),
            nn.ConvTranspose3d(decoder_features, decoder_features // 2, 2, 2),
            self._conv_block(decoder_features // 2, decoder_features // 2),
        )
        
        # Feature dimensions
        encoder_out_features = base_features * 16
        decoder_out_features = decoder_features // 2
        
        # Project features to d_model
        self.feature_proj = nn.Linear(decoder_out_features, d_model)
        self.global_proj = nn.Linear(encoder_out_features, d_model)
        
        # Vessel-masked pooling
        self.vessel_pooling = VesselMaskedPoolingLayer(num_locations)
        
        # Location-Aware Transformer
        self.location_transformer = LocationAwareTransformer(
            d_model=d_model,
            nhead=8,
            num_layers=2,
        )
        
        # Feature fusion (location + global)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification heads
        # 13 location heads
        self.location_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in range(num_locations)
        ])
        
        # Aneurysm Present head
        self.presence_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        
        # Auxiliary sphere detection head
        self.aux_sphere_head = nn.Sequential(
            nn.Conv3d(decoder_out_features, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1),
        )
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create conv block."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        vessel_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input ROI volume (B, 1, D, H, W)
            vessel_mask: Vessel segmentation mask (B, D, H, W)
            
        Returns:
            Dictionary with:
            - 'locations': (B, 13) logits for each location
            - 'presence': (B, 1) logit for Aneurysm Present
            - 'aux_sphere': (B, 1, D/4, H/4, W/4) auxiliary sphere prediction
        """
        # Encoder
        encoder_features = []
        for i, enc_block in enumerate(self.encoder):
            x = enc_block(x)
            encoder_features.append(x)
            if i < len(self.encoder) - 1:
                x = self.pool(x)
        
        # Global features from encoder
        global_feat = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        global_feat = self.global_proj(global_feat)  # (B, d_model)
        
        # Decoder
        decoder_out = self.decoder(x)  # (B, C, D', H', W')
        
        # Auxiliary sphere prediction
        aux_sphere = self.aux_sphere_head(decoder_out)
        
        # Location-specific features
        if vessel_mask is not None:
            # Vessel-masked pooling
            decoder_features = decoder_out.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
            decoder_features = decoder_features.mean(dim=(1, 2, 3))  # Simplified: (B, C)
            decoder_features = self.feature_proj(decoder_features)  # (B, d_model)
            
            # Get location features using vessel mask
            location_features = self.vessel_pooling(decoder_out, vessel_mask)  # (B, 13, C)
            
            # Project to d_model
            location_features = self.feature_proj(
                location_features.view(-1, decoder_out.shape[1])
            ).view(-1, self.num_locations, self.d_model)  # (B, 13, d_model)
        else:
            # Without mask, use spatial regions
            decoder_features = self.feature_proj(
                F.adaptive_avg_pool3d(decoder_out, 1).squeeze(-1).squeeze(-1).squeeze(-1)
            )
            location_features = decoder_features.unsqueeze(1).expand(-1, self.num_locations, -1)
        
        # Location-Aware Transformer
        location_features = self.location_transformer(location_features)
        
        # Fuse with global features
        global_expanded = global_feat.unsqueeze(1).expand(-1, self.num_locations, -1)
        fused = self.fusion(torch.cat([location_features, global_expanded], dim=-1))
        
        # Classification
        location_logits = torch.cat([
            head(fused[:, i]) for i, head in enumerate(self.location_heads)
        ], dim=-1)  # (B, 13)
        
        # Aneurysm Present (using global vessel features)
        presence_logit = self.presence_head(global_feat)  # (B, 1)
        
        return {
            'locations': location_logits,
            'presence': presence_logit.squeeze(-1),
            'aux_sphere': aux_sphere,
        }


def create_roi_classifier(config: dict = None) -> ROIClassifier:
    """Factory function to create ROI Classifier."""
    if config is None:
        config = {}
    
    return ROIClassifier(
        in_channels=config.get('in_channels', 1),
        base_features=config.get('base_features', 32),
        num_locations=config.get('num_locations', 13),
        d_model=config.get('d_model', 256),
        dropout=config.get('dropout', 0.3),
    )


if __name__ == '__main__':
    print("=" * 60)
    print("ROI CLASSIFIER TEST")
    print("=" * 60)
    
    # Create model
    model = ROIClassifier(
        in_channels=1,
        base_features=32,
        num_locations=13,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 64, 128, 128)
    vessel_mask = torch.randint(0, 14, (2, 64, 128, 128))
    
    print(f"\nInput shape: {x.shape}")
    print(f"Vessel mask shape: {vessel_mask.shape}")
    
    with torch.no_grad():
        outputs = model(x, vessel_mask)
    
    print(f"\nOutputs:")
    print(f"  Locations: {outputs['locations'].shape}")
    print(f"  Presence: {outputs['presence'].shape}")
    print(f"  Aux sphere: {outputs['aux_sphere'].shape}")
    
    print("\n✅ ROI Classifier working!")
