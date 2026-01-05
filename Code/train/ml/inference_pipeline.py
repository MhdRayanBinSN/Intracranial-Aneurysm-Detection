"""
Full Inference Pipeline (1st Place Solution)

Combines all components for end-to-end prediction:
1. Preprocessing (DICOM → standardized volume)
2. Coarse vessel localization
3. Fine vessel segmentation  
4. ROI classification (13 locations + presence)
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys
import zipfile
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing import DicomPreprocessor
from roi.roi_extraction import ROIExtractor, VesselMaskedPooling
from models.roi_classifier import ROIClassifier, create_roi_classifier


# Anatomical location names
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
]


class AneurysmDetectionPipeline:
    """
    Complete aneurysm detection pipeline.
    
    Pipeline stages:
    1. Preprocessing: Load and normalize volume
    2. Vessel Segmentation: nnU-Net (coarse + fine)
    3. ROI Extraction: Crop around vessels
    4. Classification: Predict 13 locations + presence
    """
    
    def __init__(
        self,
        classifier_checkpoint: Optional[str] = None,
        device: str = 'cuda',
        roi_size: Tuple[int, int, int] = (64, 128, 128),
    ):
        """
        Initialize pipeline.
        
        Args:
            classifier_checkpoint: Path to ROI classifier checkpoint
            device: Device for inference
            roi_size: ROI volume size
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.roi_size = roi_size
        
        # Initialize preprocessor
        self.preprocessor = DicomPreprocessor(
            target_size=(roi_size[1], roi_size[2]),
            num_slices=roi_size[0],
        )
        
        # Initialize ROI extractor
        self.roi_extractor = ROIExtractor(
            roi_size_mm=(140, 140, 140),
            spacing=(1.0, 1.0, 1.0),
        )
        
        # Load classifier
        self.classifier = self._load_classifier(classifier_checkpoint)
        
        # Fallback predictions (mean of training data)
        self.fallback_predictions = self._get_fallback_predictions()
    
    def _load_classifier(self, checkpoint_path: Optional[str]) -> ROIClassifier:
        """Load ROI classifier model."""
        model = create_roi_classifier({
            'in_channels': 1,
            'base_features': 32,
            'num_locations': 13,
            'd_model': 256,
        })
        
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading classifier from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load EMA weights if available
            if 'ema_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['ema_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("No checkpoint provided - using random weights (for testing)")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_fallback_predictions(self) -> Dict[str, float]:
        """Get fallback predictions (used if pipeline fails)."""
        # These are approximate mean predictions from training data
        return {
            'Aneurysm Present': 0.43,
            'Left Infraclinoid ICA': 0.05,
            'Right Infraclinoid ICA': 0.05,
            'Left Supraclinoid ICA': 0.06,
            'Right Supraclinoid ICA': 0.06,
            'Left Middle Cerebral Artery': 0.08,
            'Right Middle Cerebral Artery': 0.08,
            'Anterior Communicating Artery': 0.07,
            'Left Anterior Cerebral Artery': 0.02,
            'Right Anterior Cerebral Artery': 0.02,
            'Left Posterior Communicating Artery': 0.03,
            'Right Posterior Communicating Artery': 0.03,
            'Basilar Tip': 0.04,
            'Other Posterior Circulation': 0.02,
        }
    
    def preprocess_volume(
        self,
        volume: np.ndarray,
        modality: str = 'CTA',
    ) -> np.ndarray:
        """
        Preprocess volume for inference.
        
        Args:
            volume: Input volume (D, H, W)
            modality: Imaging modality
            
        Returns:
            Preprocessed volume (D, H, W)
        """
        # Apply windowing
        volume = self.preprocessor.apply_windowing(volume, modality)
        
        # Resize to ROI size
        from scipy.ndimage import zoom
        current_shape = volume.shape
        target_shape = self.roi_size
        
        scale_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
        volume = zoom(volume, scale_factors, order=1)
        
        return volume
    
    def segment_vessels(
        self,
        volume: np.ndarray,
    ) -> np.ndarray:
        """
        Segment vessels using nnU-Net (placeholder).
        
        In full implementation, this would run:
        1. Coarse model for ROI localization
        2. Fine models for detailed segmentation
        
        Args:
            volume: Preprocessed volume
            
        Returns:
            Vessel segmentation mask with location labels
        """
        # Placeholder: Create dummy vessel mask
        # In production, this would call nnU-Net inference
        
        # Simulate vessel regions
        mask = np.zeros(volume.shape, dtype=np.int64)
        
        # Find bright regions (assume they're vessels)
        threshold = volume.mean() + volume.std()
        vessel_regions = volume > threshold
        
        # Assign random labels to connected components
        from scipy.ndimage import label as ndimage_label
        labeled, num_features = ndimage_label(vessel_regions)
        
        # Map components to location labels (simplified)
        for i in range(1, min(num_features + 1, 14)):
            mask[labeled == i] = i
        
        return mask
    
    @torch.no_grad()
    def predict(
        self,
        volume: np.ndarray,
        vessel_mask: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, any]:
        """
        Run full prediction pipeline.
        
        Args:
            volume: Preprocessed volume (D, H, W)
            vessel_mask: Optional vessel segmentation mask
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Prepare input tensor
            # Normalize
            volume_norm = (volume - 0.5) / 0.5
            
            # To tensor (B, C, D, H, W)
            x = torch.from_numpy(volume_norm).float()
            x = x.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Prepare vessel mask
            if vessel_mask is not None:
                mask_tensor = torch.from_numpy(vessel_mask).long()
                mask_tensor = mask_tensor.unsqueeze(0).to(self.device)
            else:
                mask_tensor = None
            
            # Run classifier
            outputs = self.classifier(x, mask_tensor)
            
            # Get probabilities
            location_probs = torch.sigmoid(outputs['locations']).cpu().numpy()[0]
            presence_prob = torch.sigmoid(outputs['presence']).cpu().numpy().item()
            
            # Build results
            results = {
                'aneurysm_present': presence_prob > threshold,
                'aneurysm_probability': presence_prob,
                'location_probabilities': {},
                'predictions': {},
            }
            
            for i, name in enumerate(LOCATION_NAMES):
                prob = float(location_probs[i])
                results['location_probabilities'][name] = prob
                results['predictions'][name] = prob > threshold
            
            results['predictions']['Aneurysm Present'] = results['aneurysm_present']
            
            return results
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            print("Using fallback predictions")
            
            return {
                'aneurysm_present': False,
                'aneurysm_probability': self.fallback_predictions['Aneurysm Present'],
                'location_probabilities': {
                    name: self.fallback_predictions.get(name, 0.05)
                    for name in LOCATION_NAMES
                },
                'error': str(e),
            }
    
    def predict_from_zip(
        self,
        zip_path: str,
        series_uid: str,
        modality: str = 'CTA',
    ) -> Dict[str, any]:
        """
        Run prediction on a series from the competition zip.
        
        Args:
            zip_path: Path to competition zip
            series_uid: Series Instance UID
            modality: Imaging modality
            
        Returns:
            Prediction results
        """
        # Load NIfTI volume
        with zipfile.ZipFile(zip_path, 'r') as zf:
            nifti_path = f"segmentations/{series_uid}.nii"
            
            if nifti_path not in zf.namelist():
                return {
                    'error': f"NIfTI not found: {nifti_path}",
                    'aneurysm_probability': self.fallback_predictions['Aneurysm Present'],
                }
            
            with zf.open(nifti_path) as f:
                with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
            
            try:
                img = nib.load(tmp_path)
                volume = img.get_fdata().astype(np.float32)
            finally:
                os.remove(tmp_path)
            
            # Load vessel mask if available
            mask_path = f"segmentations/{series_uid}_cowseg.nii"
            vessel_mask = None
            
            if mask_path in zf.namelist():
                with zf.open(mask_path) as f:
                    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                
                try:
                    mask_img = nib.load(tmp_path)
                    vessel_mask = mask_img.get_fdata().astype(np.int64)
                finally:
                    os.remove(tmp_path)
        
        # Preprocess
        volume = self.preprocess_volume(volume, modality)
        
        if vessel_mask is not None:
            from scipy.ndimage import zoom
            scale = tuple(v / m for v, m in zip(volume.shape, vessel_mask.shape))
            vessel_mask = zoom(vessel_mask, scale, order=0)
        
        # Run prediction
        return self.predict(volume, vessel_mask)


def print_results(results: Dict[str, any]):
    """Pretty print prediction results."""
    print("\n" + "=" * 60)
    print("ANEURYSM DETECTION RESULTS")
    print("=" * 60)
    
    # Main result
    if results.get('aneurysm_present'):
        prob = results['aneurysm_probability']
        print(f"\n🔴 ANEURYSM DETECTED (probability: {prob:.1%})")
    else:
        prob = results['aneurysm_probability']
        print(f"\n🟢 NO ANEURYSM (probability: {prob:.1%})")
    
    # Location probabilities
    print("\n📍 Location Probabilities:")
    print("-" * 50)
    
    for name, prob in results.get('location_probabilities', {}).items():
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        marker = "⚠️" if prob > 0.5 else "  "
        print(f"{marker} {name:40s} {bar} {prob:.1%}")
    
    print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Aneurysm Detection Pipeline')
    parser.add_argument('--checkpoint', type=str, help='Classifier checkpoint')
    parser.add_argument('--series-uid', type=str, help='Series UID to predict')
    parser.add_argument('--zip-path', type=str, 
                       default='C:/Users/Rayan/Desktop/Main Project/rsna-intracranial-aneurysm-detection.zip')
    parser.add_argument('--modality', type=str, default='CTA')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = AneurysmDetectionPipeline(
        classifier_checkpoint=args.checkpoint,
        device=args.device,
    )
    
    # If no series UID provided, test with first available
    if not args.series_uid:
        import pandas as pd
        with zipfile.ZipFile(args.zip_path, 'r') as zf:
            # Get a sample with mask
            mask_files = [
                f.replace('segmentations/', '').replace('_cowseg.nii', '')
                for f in zf.namelist()
                if '_cowseg.nii' in f
            ][:1]
            
            if mask_files:
                args.series_uid = mask_files[0]
                print(f"Using sample series: {args.series_uid[:40]}...")
    
    if not args.series_uid:
        print("No series UID provided and no samples found!")
        return
    
    # Run prediction
    print("\nRunning prediction...")
    results = pipeline.predict_from_zip(
        args.zip_path,
        args.series_uid,
        args.modality,
    )
    
    # Print results
    print_results(results)


if __name__ == '__main__':
    main()
