"""
Inference Script for Aneurysm Detection + Segmentation.

Use this script AFTER training to make predictions on new brain scans.

Usage:
    python inference.py --checkpoint checkpoints/best_multitask.pth --input patient_scan.nii
    python inference.py --checkpoint checkpoints/best_multitask.pth --series-uid 1.2.826.xxx
"""

import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models import MultiTaskUNet
from data.preprocessing import DicomPreprocessor
import zipfile


# Location names for output
LOCATION_NAMES = [
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
    'Aneurysm Present',  # Summary column
]


def load_model(checkpoint_path: str, device: str = 'cuda') -> MultiTaskUNet:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: 'cuda' or 'cpu'
        
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Create model
    model = MultiTaskUNet(
        in_channels=config.get('model', {}).get('in_channels', 1),
        num_classes=config.get('model', {}).get('num_classes', 14),
        base_features=config.get('model', {}).get('base_features', 32),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model


def preprocess_scan(
    series_path: str,
    zip_path: str = None,
    modality: str = 'CTA',
) -> torch.Tensor:
    """
    Load and preprocess a brain scan.
    
    Args:
        series_path: Path to DICOM series or SeriesInstanceUID
        zip_path: Path to competition zip file (if loading from zip)
        modality: 'CTA', 'MRA', or 'MRI'
        
    Returns:
        Preprocessed tensor of shape (1, 1, D, H, W)
    """
    preprocessor = DicomPreprocessor(
        target_size=(128, 128),
        num_slices=32,
    )
    
    if zip_path:
        # Load from zip file
        with zipfile.ZipFile(zip_path, 'r') as zf:
            volume, metadata = preprocessor.preprocess(
                f"series/{series_path}",
                modality,
                zf
            )
    else:
        # Load from directory
        volume, metadata = preprocessor.preprocess(series_path, modality)
    
    # Normalize
    volume = (volume - 0.5) / 0.5
    
    # Convert to tensor: (D, H, W) -> (1, 1, D, H, W)
    tensor = torch.from_numpy(volume).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    return tensor


@torch.no_grad()
def predict(
    model: MultiTaskUNet,
    volume: torch.Tensor,
    device: str = 'cuda',
    threshold: float = 0.5,
) -> dict:
    """
    Make prediction on a brain scan.
    
    Args:
        model: Trained MultiTaskUNet
        volume: Preprocessed tensor (1, 1, D, H, W)
        device: 'cuda' or 'cpu'
        threshold: Probability threshold for positive prediction
        
    Returns:
        Dictionary with:
            - 'aneurysm_present': bool
            - 'probabilities': dict mapping location to probability
            - 'segmentation_mask': numpy array (D, H, W)
    """
    volume = volume.to(device)
    
    # Forward pass
    outputs = model(volume)
    
    # Get classification probabilities
    probs = torch.sigmoid(outputs['classification']).cpu().numpy()[0]
    
    # Get segmentation mask
    seg_mask = torch.sigmoid(outputs['segmentation']).cpu().numpy()[0, 0]
    seg_mask = (seg_mask > threshold).astype(np.uint8)
    
    # Create results dictionary
    results = {
        'aneurysm_present': probs[13] > threshold,
        'aneurysm_probability': float(probs[13]),
        'probabilities': {},
        'segmentation_mask': seg_mask,
        'mask_volume_voxels': int(seg_mask.sum()),
    }
    
    # Add per-location probabilities
    for i, name in enumerate(LOCATION_NAMES):
        results['probabilities'][name] = float(probs[i])
    
    return results


def print_results(results: dict):
    """Print prediction results in a nice format."""
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    # Main result
    if results['aneurysm_present']:
        print(f"\n🔴 ANEURYSM DETECTED (probability: {results['aneurysm_probability']:.1%})")
    else:
        print(f"\n🟢 NO ANEURYSM (probability: {results['aneurysm_probability']:.1%})")
    
    # Location probabilities
    print("\n📍 Location Probabilities:")
    print("-" * 40)
    
    for name, prob in results['probabilities'].items():
        if name == 'Aneurysm Present':
            continue  # Skip summary, already shown above
        
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        marker = "⚠️" if prob > 0.5 else "  "
        print(f"{marker} {name:35s} {bar} {prob:.1%}")
    
    # Segmentation info
    print(f"\n🎯 Segmentation Mask:")
    print(f"   Volume: {results['mask_volume_voxels']} voxels")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Aneurysm Detection Inference')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth)',
    )
    parser.add_argument(
        '--series-uid',
        type=str,
        help='SeriesInstanceUID to load from zip',
    )
    parser.add_argument(
        '--zip-path',
        type=str,
        default='C:/Users/Rayan/Desktop/Main Project/rsna-intracranial-aneurysm-detection.zip',
        help='Path to competition zip file',
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='CTA',
        choices=['CTA', 'MRA', 'MRI'],
        help='Imaging modality',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Probability threshold',
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nYou need to train the model first!")
        print("Run: python train_multitask.py")
        sys.exit(1)
    
    # Load model
    model = load_model(args.checkpoint, args.device)
    
    # Load and preprocess scan
    print(f"\nLoading scan: {args.series_uid}")
    volume = preprocess_scan(
        args.series_uid,
        args.zip_path,
        args.modality,
    )
    print(f"Scan shape: {volume.shape}")
    
    # Make prediction
    print("\nRunning inference...")
    results = predict(model, volume, args.device, args.threshold)
    
    # Print results
    print_results(results)


if __name__ == '__main__':
    main()
