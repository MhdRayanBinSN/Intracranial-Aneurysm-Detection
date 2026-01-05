"""
Multi-label classification head with optional attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on relevant regions."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()
        
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.fc(combined).view(b, c, 1, 1, 1)
        
        return x * attention.expand_as(x)


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class MultiLabelClassifier(nn.Module):
    """
    Multi-label classification head for aneurysm detection.
    
    Supports:
    - Optional attention mechanism
    - Dropout for regularization
    - Multiple hidden layers
    """
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 14,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        """
        Initialize the classifier.
        
        Args:
            in_features: Number of input features from backbone
            num_classes: Number of output classes (14 for this task)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Build MLP layers
        layers = []
        prev_dim = in_features
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),  # LayerNorm works with batch_size=1
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Final classification layer
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (B, in_features)
            
        Returns:
            logits: Output logits of shape (B, num_classes)
        """
        x = self.mlp(x)
        logits = self.classifier(x)
        return logits


class AnatomyAwareClassifier(nn.Module):
    """
    Anatomy-aware multi-label classifier.
    
    Groups predictions by anatomical regions for better interpretability:
    - Left/Right Infraclinoid Internal Carotid Artery
    - Left/Right Supraclinoid Internal Carotid Artery
    - Left/Right Middle Cerebral Artery
    - Left/Right Anterior Cerebral Artery
    - Left/Right Posterior Communicating Artery
    - Basilar Tip
    - Other Posterior Circulation
    - Overall Presence
    """
    
    # Anatomical region groups
    REGIONS = {
        'anterior_circulation': [0, 1, 2, 3, 4, 5, 6, 7],  # ICA + MCA + ACA
        'posterior_circulation': [8, 9, 10, 11],  # PComm + Basilar + Other
        'overall': [12],  # Aneurysm Present
    }
    
    def __init__(
        self,
        in_features: int,
        num_classes: int = 14,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),  # LayerNorm works with batch_size=1
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Region-specific heads
        self.anterior_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 8),  # 8 anterior locations
        )
        
        self.posterior_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4),  # 4 posterior locations
        )
        
        self.overall_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # Overall presence
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x)
        
        anterior = self.anterior_head(shared)
        posterior = self.posterior_head(shared)
        overall = self.overall_head(shared)
        
        # Concatenate all predictions
        logits = torch.cat([anterior, posterior, overall], dim=1)
        
        return logits


if __name__ == '__main__':
    # Test classifiers
    print("Testing classification heads...")
    
    # Create dummy features
    features = torch.randn(4, 512)
    
    # Test MultiLabelClassifier
    classifier = MultiLabelClassifier(in_features=512, num_classes=14)
    out = classifier(features)
    print(f"MultiLabelClassifier: input {features.shape} -> output {out.shape}")
    
    # Test AnatomyAwareClassifier
    anatomy_classifier = AnatomyAwareClassifier(in_features=512, num_classes=14)
    out = anatomy_classifier(features)
    print(f"AnatomyAwareClassifier: input {features.shape} -> output {out.shape}")
    
    print("All classifiers working correctly!")
