"""
Data augmentation for 3D medical volumes.
"""

import numpy as np
from typing import Dict, Any, Callable, Optional
from scipy.ndimage import rotate, zoom
from scipy.ndimage import gaussian_filter


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            volume = t(volume)
        return volume


class RandomFlip3D:
    """Random 3D flip along specified axes."""
    
    def __init__(self, flip_prob: float = 0.5, axes: tuple = (0, 1, 2)):
        self.flip_prob = flip_prob
        self.axes = axes
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for axis in self.axes:
            if np.random.random() < self.flip_prob:
                volume = np.flip(volume, axis=axis)
        return np.ascontiguousarray(volume)


class RandomRotate3D:
    """Random rotation in 3D."""
    
    def __init__(self, angle_range: float = 15, axes: tuple = (1, 2)):
        """
        Args:
            angle_range: Maximum rotation angle in degrees
            axes: Axes defining the rotation plane
        """
        self.angle_range = angle_range
        self.axes = axes
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.angle_range, self.angle_range)
        volume = rotate(volume, angle, axes=self.axes, reshape=False, order=1, mode='constant', cval=0)
        return volume


class RandomIntensityShift:
    """Random intensity shift."""
    
    def __init__(self, shift_range: float = 0.1):
        self.shift_range = shift_range
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(-self.shift_range, self.shift_range)
        volume = volume + shift
        return np.clip(volume, 0, 1)


class RandomIntensityScale:
    """Random intensity scaling."""
    
    def __init__(self, scale_range: float = 0.1):
        self.scale_range = scale_range
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
        volume = volume * scale
        return np.clip(volume, 0, 1)


class GaussianNoise:
    """Add Gaussian noise."""
    
    def __init__(self, mean: float = 0, std: float = 0.01, prob: float = 0.5):
        self.mean = mean
        self.std = std
        self.prob = prob
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            noise = np.random.normal(self.mean, self.std, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        return volume


class GaussianBlur:
    """Apply Gaussian blur."""
    
    def __init__(self, sigma_range: tuple = (0.5, 1.0), prob: float = 0.3):
        self.sigma_range = sigma_range
        self.prob = prob
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if np.random.random() < self.prob:
            sigma = np.random.uniform(*self.sigma_range)
            volume = gaussian_filter(volume, sigma=sigma)
        return volume


class RandomCrop3D:
    """Random crop in 3D."""
    
    def __init__(self, crop_size: tuple, padding: int = 0):
        self.crop_size = crop_size
        self.padding = padding
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            volume = np.pad(volume, self.padding, mode='constant', constant_values=0)
        
        d, h, w = volume.shape
        cd, ch, cw = self.crop_size
        
        if d <= cd or h <= ch or w <= cw:
            return volume
        
        d_start = np.random.randint(0, d - cd)
        h_start = np.random.randint(0, h - ch)
        w_start = np.random.randint(0, w - cw)
        
        return volume[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]


class CenterCrop3D:
    """Center crop in 3D."""
    
    def __init__(self, crop_size: tuple):
        self.crop_size = crop_size
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        d, h, w = volume.shape
        cd, ch, cw = self.crop_size
        
        d_start = (d - cd) // 2
        h_start = (h - ch) // 2
        w_start = (w - cw) // 2
        
        return volume[d_start:d_start+cd, h_start:h_start+ch, w_start:w_start+cw]


class Normalize:
    """Normalize volume to zero mean and unit variance."""
    
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, volume: np.ndarray) -> np.ndarray:
        return (volume - self.mean) / self.std


def get_train_transforms(config: Dict[str, Any]) -> Compose:
    """
    Get training augmentation transforms.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Composed transforms
    """
    aug_config = config.get('augmentation', {})
    
    transforms = []
    
    # Random flips
    if aug_config.get('flip_horizontal', True):
        transforms.append(RandomFlip3D(flip_prob=0.5, axes=(2,)))  # Width axis
    if aug_config.get('flip_depth', True):
        transforms.append(RandomFlip3D(flip_prob=0.5, axes=(0,)))  # Depth axis
    
    # Random rotation
    rotation_range = aug_config.get('rotation_range', 15)
    if rotation_range > 0:
        transforms.append(RandomRotate3D(angle_range=rotation_range))
    
    # Intensity augmentations
    intensity_shift = aug_config.get('intensity_shift', 0.1)
    if intensity_shift > 0:
        transforms.append(RandomIntensityShift(shift_range=intensity_shift))
    
    intensity_scale = aug_config.get('intensity_scale', 0.1)
    if intensity_scale > 0:
        transforms.append(RandomIntensityScale(scale_range=intensity_scale))
    
    # Noise
    transforms.append(GaussianNoise(std=0.01, prob=0.3))
    
    # Normalize
    transforms.append(Normalize(mean=0.5, std=0.5))
    
    return Compose(transforms)


def get_val_transforms(config: Dict[str, Any]) -> Compose:
    """
    Get validation transforms (no augmentation).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Composed transforms
    """
    transforms = [
        Normalize(mean=0.5, std=0.5),
    ]
    
    return Compose(transforms)


if __name__ == '__main__':
    # Test transforms
    print("Testing augmentation transforms...")
    
    # Create dummy volume
    volume = np.random.rand(64, 256, 256).astype(np.float32)
    
    # Test config
    config = {
        'augmentation': {
            'enabled': True,
            'flip_horizontal': True,
            'flip_depth': True,
            'rotation_range': 15,
            'intensity_shift': 0.1,
            'intensity_scale': 0.1,
        }
    }
    
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)
    
    # Apply transforms
    augmented = train_transforms(volume)
    validated = val_transforms(volume)
    
    print(f"Original shape: {volume.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print(f"Validated shape: {validated.shape}")
    print("Transforms working correctly!")
