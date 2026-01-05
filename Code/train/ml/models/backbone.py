"""
3D CNN Backbone models for medical image classification.
Implements ResNet3D and EfficientNet3D variants.
"""

import torch
import torch.nn as nn
from typing import Optional, List
import math


class Conv3DBlock(nn.Module):
    """Basic 3D convolution block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock3D(nn.Module):
    """3D Residual block for ResNet3D."""
    
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BottleneckBlock3D(nn.Module):
    """3D Bottleneck block for deeper ResNet3D."""
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet3D(nn.Module):
    """
    3D ResNet for volumetric medical image classification.
    
    Variants:
    - resnet3d_18: [2, 2, 2, 2] with BasicBlock
    - resnet3d_34: [3, 4, 6, 3] with BasicBlock
    - resnet3d_50: [3, 4, 6, 3] with Bottleneck
    """
    
    def __init__(
        self,
        block: type,
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 14,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        
        self.in_planes = 64
        
        # Initial convolution
        self.conv1 = nn.Conv3d(
            in_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Output feature dimension
        self.num_features = 512 * block.expansion
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock3D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResidualBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            features: Feature tensor of shape (B, num_features)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def resnet3d_18(in_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet3D-18 model."""
    return ResNet3D(ResidualBlock3D, [2, 2, 2, 2], in_channels=in_channels, **kwargs)


def resnet3d_34(in_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet3D-34 model."""
    return ResNet3D(ResidualBlock3D, [3, 4, 6, 3], in_channels=in_channels, **kwargs)


def resnet3d_50(in_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet3D-50 model."""
    return ResNet3D(BottleneckBlock3D, [3, 4, 6, 3], in_channels=in_channels, **kwargs)


# Model registry
BACKBONES = {
    'resnet3d_18': resnet3d_18,
    'resnet3d_34': resnet3d_34,
    'resnet3d_50': resnet3d_50,
}


def get_backbone(name: str, in_channels: int = 1, **kwargs) -> nn.Module:
    """
    Get backbone model by name.
    
    Args:
        name: Model name
        in_channels: Number of input channels
        **kwargs: Additional arguments
        
    Returns:
        Backbone model
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONES.keys())}")
    
    return BACKBONES[name](in_channels=in_channels, **kwargs)


if __name__ == '__main__':
    # Test backbone
    print("Testing ResNet3D backbones...")
    
    # Create dummy input (B, C, D, H, W)
    x = torch.randn(2, 1, 64, 256, 256)
    
    for name in BACKBONES:
        model = get_backbone(name)
        out = model(x)
        print(f"{name}: input {x.shape} -> output {out.shape}")
    
    print("All backbones working correctly!")
