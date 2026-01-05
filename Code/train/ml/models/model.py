"""
Full model architecture combining backbone and classifier.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .backbone import get_backbone, ResNet3D
from .classifier import MultiLabelClassifier, AnatomyAwareClassifier, SEBlock3D


class AneurysmDetector(nn.Module):
    """
    Full model for Intracranial Aneurysm Detection.
    
    Combines:
    - 3D CNN backbone (ResNet3D variants)
    - Optional attention mechanism
    - Multi-label classification head
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet3d_18',
        num_classes: int = 14,
        in_channels: int = 1,
        dropout: float = 0.3,
        use_attention: bool = True,
        pretrained: bool = False,
    ):
        """
        Initialize the model.
        
        Args:
            backbone_name: Name of the backbone ('resnet3d_18', 'resnet3d_34', 'resnet3d_50')
            num_classes: Number of output classes (14)
            in_channels: Number of input channels (1 for grayscale CT/MR)
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            pretrained: Whether to load pretrained weights (if available)
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Create backbone
        self.backbone = get_backbone(backbone_name, in_channels=in_channels)
        num_features = self.backbone.num_features
        
        # Optional attention module
        if use_attention:
            self.attention = SEBlock3D(num_features, reduction=16)
        else:
            self.attention = None
        
        # Global pooling (already in backbone, but keep for flexibility)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.classifier = MultiLabelClassifier(
            in_features=num_features,
            num_classes=num_classes,
            hidden_dims=[512, 256],
            dropout=dropout,
            use_attention=use_attention,
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Dictionary with 'logits' and 'probabilities'
        """
        # Extract features through backbone
        # Note: backbone already includes global pooling
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Get probabilities
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'features': features,
        }
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make predictions with thresholding.
        
        Args:
            x: Input tensor
            threshold: Probability threshold for positive prediction
            
        Returns:
            Binary predictions tensor
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predictions = (output['probabilities'] > threshold).float()
        return predictions
    
    def get_attention_maps(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention maps if available
        """
        if not self.use_attention:
            return None
        
        # This would need a more sophisticated implementation
        # for proper Grad-CAM style visualization
        return None


class AneurysmDetectorLightning(nn.Module):
    """
    PyTorch Lightning compatible wrapper for training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        model_config = config.get('model', {})
        
        self.model = AneurysmDetector(
            backbone_name=model_config.get('name', 'resnet3d_18'),
            num_classes=model_config.get('num_classes', 14),
            in_channels=1,
            dropout=model_config.get('dropout', 0.3),
            use_attention=model_config.get('use_attention', True),
            pretrained=model_config.get('pretrained', False),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model(x)


def create_model(config: Dict[str, Any]) -> AneurysmDetector:
    """
    Factory function to create model from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    
    model = AneurysmDetector(
        backbone_name=model_config.get('name', 'resnet3d_18'),
        num_classes=model_config.get('num_classes', 14),
        in_channels=1,
        dropout=model_config.get('dropout', 0.3),
        use_attention=model_config.get('use_attention', True),
        pretrained=model_config.get('pretrained', False),
    )
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test full model
    print("Testing AneurysmDetector model...")
    
    # Create model
    model = AneurysmDetector(
        backbone_name='resnet3d_18',
        num_classes=14,
        use_attention=True,
    )
    
    # Print parameter count
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 64, 256, 256)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probabilities shape: {output['probabilities'].shape}")
    print(f"Features shape: {output['features'].shape}")
    
    # Test prediction
    predictions = model.predict(x)
    print(f"Predictions shape: {predictions.shape}")
    
    print("\nModel working correctly!")
