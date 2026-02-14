"""
Advanced preprocessing module based on Kaggle Winning Solutions.

Implements techniques from RSNA Intracranial Aneurysm Detection competition:
- Multi-window CT visualization (soft tissue, bone, vessel)
- Skull stripping using morphological operations
- Vessel enhancement with Frangi/Sato filters
- CLAHE contrast enhancement
- Proper HU windowing for CTA

Reference: Kaggle 1st place solution by team "tomoon33"
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from typing import Tuple, Optional, Dict
import warnings

# Try to import optional dependencies
try:
    from skimage import filters, morphology, exposure, measure
    from skimage.filters import frangi, sato
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available. Some features disabled.")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV not available. Some features disabled.")


class KaggleWinningPreprocessor:
    """
    Advanced preprocessing pipeline based on Kaggle winning solutions.
    
    Key features:
    1. Multi-window HU visualization (3-channel RGB from CT)
    2. Skull stripping to focus on brain/vessels
    3. Vessel enhancement using Frangi/Sato filters
    4. CLAHE for adaptive contrast enhancement
    5. Proper normalization for deep learning
    """
    
    # ============ OPTIMAL WINDOW SETTINGS ============
    # Based on Kaggle winning solutions and radiology standards
    
    # Window 1: Brain parenchyma (soft tissue)
    BRAIN_WINDOW = {'center': 40, 'width': 80}
    
    # Window 2: Blood/hemorrhage (subdural)
    BLOOD_WINDOW = {'center': 75, 'width': 200}
    
    # Window 3: Bone/calcification
    BONE_WINDOW = {'center': 600, 'width': 2800}
    
    # Window 4: CTA vessels (contrast enhanced)
    VESSEL_WINDOW = {'center': 300, 'width': 600}
    
    # Window 5: Stroke window (broad)
    STROKE_WINDOW = {'center': 40, 'width': 400}
    
    @staticmethod
    def detect_modality(image: np.ndarray, metadata: Optional[Dict] = None) -> str:
        """
        Detect if image is CT or MRI based on metadata or pixel intensity.
        
        Args:
            image: Raw pixel array
            metadata: DICOM metadata dictionary
            
        Returns:
            'CT' or 'MRI'
        """
        # 1. Trust metadata if available
        if metadata and 'Modality' in metadata:
            mod = str(metadata['Modality']).upper()
            if 'CT' in mod: return 'CT'
            if 'MR' in mod: return 'MRI'
            
        # 2. Heuristic based on value range
        # CT values start from -1000 (Air)
        # MRI values are strictly non-negative (usually)
        min_val = np.min(image)
        if min_val < -50:
            return 'CT'
        else:
            # Check for typical MRI range (usually 0-4096 or 0-255)
            # CT range usually extends to 3000 (Bone)
            return 'MRI'
    
    def __init__(
        self,
        use_multi_window: bool = True,
        use_skull_strip: bool = True,
        use_vessel_enhance: bool = True,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        vessel_scales: Tuple = (1, 2, 3),
    ):
        """
        Initialize advanced preprocessor.
        
        Args:
            use_multi_window: Create 3-channel from different CT windows
            use_skull_strip: Remove skull using morphological operations
            use_vessel_enhance: Apply Frangi filter for vessel enhancement
            use_clahe: Apply CLAHE contrast enhancement
            clahe_clip_limit: CLAHE clip limit for contrast
            vessel_scales: Scales for Frangi vessel filter
        """
        self.use_multi_window = use_multi_window
        self.use_skull_strip = use_skull_strip
        self.use_vessel_enhance = use_vessel_enhance
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.vessel_scales = vessel_scales
    
    def apply_window(
        self, 
        image: np.ndarray, 
        center: float, 
        width: float
    ) -> np.ndarray:
        """
        Apply CT windowing to convert HU values to displayable range.
        
        Args:
            image: Input image in HU values
            center: Window center (level)
            width: Window width
            
        Returns:
            Normalized image in [0, 1] range
        """
        lower = center - width / 2
        upper = center + width / 2
        
        windowed = np.clip(image, lower, upper)
        windowed = (windowed - lower) / (width + 1e-8)
        
        return windowed.astype(np.float32)
    
    def create_multi_window_image(
        self, 
        image: np.ndarray
    ) -> np.ndarray:
        """
        Create 3-channel image from different CT windows.
        This is a KEY technique from Kaggle winning solutions.
        
        Channel 0 (Red): Brain/soft tissue window
        Channel 1 (Green): Blood/vessel window  
        Channel 2 (Blue): Stroke/wide window
        
        Args:
            image: Input image in HU values
            
        Returns:
            3-channel image, shape (H, W, 3), values in [0, 1]
        """
        # Channel 1: Brain window (soft tissue details)
        ch1 = self.apply_window(
            image, 
            self.BRAIN_WINDOW['center'], 
            self.BRAIN_WINDOW['width']
        )
        
        # Channel 2: Vessel/CTA window (contrast enhanced blood)
        ch2 = self.apply_window(
            image,
            self.VESSEL_WINDOW['center'],
            self.VESSEL_WINDOW['width']
        )
        
        # Channel 3: Stroke window (wide view)
        ch3 = self.apply_window(
            image,
            self.STROKE_WINDOW['center'],
            self.STROKE_WINDOW['width']
        )
        
        # Stack into RGB
        multi_window = np.stack([ch1, ch2, ch3], axis=-1)
        
        return multi_window

    def preprocess_mri(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess MRI image using percentile normalization.
        since MRI units are arbitrary (unlike CT HU).
        
        Args:
            image: Raw MRI image
            
        Returns:
            Normalized 3-channel RGB image
        """
        # Robust min-max normalization
        p1, p99 = np.percentile(image, (1, 99))
        img_norm = np.clip(image, p1, p99)
        img_norm = (img_norm - p1) / (p99 - p1 + 1e-8)
        img_norm = img_norm.astype(np.float32)
        
        # Apply CLAHE if enabled
        if self.use_clahe:
            img_norm = self.apply_clahe(img_norm)
            
        # For MRI, we create a pseudo-color or just grayscale 3-channel
        # Variation 1: Standard (Red)
        ch1 = img_norm
        
        # Variation 2: Gamma Correction (Green) - Darker
        ch2 = np.power(img_norm, 1.5)
        
        # Variation 3: S-Curve / Sigmoid (Blue) - Higher Contrast
        ch3 = 1 / (1 + np.exp(-10 * (img_norm - 0.5)))
        
        return np.stack([ch1, ch2, ch3], axis=-1)

    
    def skull_strip(
        self, 
        image: np.ndarray,
        hu_threshold: float = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove skull from CT image using morphological operations.
        
        Based on Kaggle winning approach:
        1. Threshold to find bone (HU > threshold)
        2. Find largest connected component (brain cavity)
        3. Fill holes and create brain mask
        
        Args:
            image: Input image in HU values
            hu_threshold: HU threshold for bone detection
            
        Returns:
            masked_image: Image with skull removed
            brain_mask: Binary mask of brain region
        """
        # Step 1: Create binary mask of non-air regions
        binary = image > -500  # Air is around -1000 HU
        
        # Step 2: Fill holes to get solid regions
        filled = binary_fill_holes(binary)
        
        # Step 3: Find the brain (central region, avoiding skull)
        # Erode to separate brain from skull
        eroded = binary_erosion(filled, iterations=5)
        
        # Step 4: Label connected components
        labeled, num_features = ndimage.label(eroded)
        
        if num_features == 0:
            # No features found, return original
            return image, np.ones_like(image, dtype=bool)
        
        # Step 5: Find the largest component (should be brain)
        sizes = ndimage.sum(eroded, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        
        brain_mask = labeled == largest_label
        
        # Step 6: Dilate back to include brain edges
        brain_mask = binary_dilation(brain_mask, iterations=7)
        brain_mask = binary_fill_holes(brain_mask)
        
        # Step 7: Apply mask
        masked_image = image.copy()
        masked_image[~brain_mask] = -1000  # Set outside to air HU
        
        return masked_image, brain_mask
    
    def enhance_vessels(
        self, 
        image: np.ndarray,
        scales: Tuple = None
    ) -> np.ndarray:
        """
        Enhance vessel structures using Frangi filter.
        
        The Frangi filter enhances tubular structures (vessels)
        by analyzing eigenvalues of the Hessian matrix.
        
        Args:
            image: Input image (normalized 0-1)
            scales: Scales for multi-scale analysis
            
        Returns:
            Vessel-enhanced image
        """
        if not HAS_SKIMAGE:
            return image
        
        if scales is None:
            scales = self.vessel_scales
        
        # Ensure image is in correct range
        img_normalized = image.astype(np.float64)
        if img_normalized.max() > 1:
            img_normalized = img_normalized / img_normalized.max()
        
        try:
            # Apply Frangi filter for tubular structure enhancement
            # black_ridges=False for bright vessels on dark background (CTA)
            vessel_enhanced = frangi(
                img_normalized,
                sigmas=scales,
                black_ridges=False,
                mode='reflect'
            )
            
            # Normalize output
            if vessel_enhanced.max() > 0:
                vessel_enhanced = vessel_enhanced / vessel_enhanced.max()
            
            return vessel_enhanced.astype(np.float32)
            
        except Exception as e:
            warnings.warn(f"Frangi filter failed: {e}")
            return image
    
    def apply_clahe(
        self, 
        image: np.ndarray,
        clip_limit: float = None
    ) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        CLAHE improves local contrast while limiting noise amplification.
        Used by many Kaggle winning solutions.
        
        Args:
            image: Input image (normalized 0-1)
            clip_limit: Clipping limit for CLAHE
            
        Returns:
            Contrast-enhanced image
        """
        if clip_limit is None:
            clip_limit = self.clahe_clip_limit
        
        if HAS_CV2:
            # OpenCV CLAHE (faster)
            img_uint8 = (image * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit, 
                tileGridSize=(8, 8)
            )
            enhanced = clahe.apply(img_uint8)
            return enhanced.astype(np.float32) / 255.0
            
        elif HAS_SKIMAGE:
            # scikit-image alternative
            return exposure.equalize_adapthist(
                image, 
                clip_limit=clip_limit / 100.0
            ).astype(np.float32)
        
        else:
            return image
    
    def preprocess_slice(
        self, 
        image: np.ndarray,
        return_channels: bool = False
    ) -> np.ndarray:
        """
        Full preprocessing pipeline for a single CT slice.
        
        Pipeline:
        1. Skull stripping (optional)
        2. Multi-window transformation (optional)
        3. Vessel enhancement (optional)
        4. CLAHE contrast enhancement (optional)
        
        Args:
            image: Input image in HU values
            return_channels: If True, return multi-channel output
            
        Returns:
            Preprocessed image
        """
        processed = image.copy().astype(np.float32)
        
        # Detect Modality if not specified (naive check not performed here, passed in or assumed CT)
        # But here we assume input 'image' is raw. 
        # Ideally, modality should be passed.
        # But for backward compatibility, we default to CT or detect.
        modality = self.detect_modality(processed)
        
        if modality == 'MRI':
            return self.preprocess_mri(processed)
        
        # CT Processing Pipeline
        
        # Step 1: Skull stripping
        if self.use_skull_strip:
            processed, brain_mask = self.skull_strip(processed)
        else:
            brain_mask = np.ones_like(processed, dtype=bool)
        
        # Step 2: Multi-window or single window
        if self.use_multi_window and return_channels:
            # Create 3-channel multi-window image
            output = self.create_multi_window_image(processed)
            
            # Apply CLAHE to each channel
            if self.use_clahe:
                for i in range(3):
                    output[:, :, i] = self.apply_clahe(output[:, :, i])
            
        else:
            # Single channel CTA window
            output = self.apply_window(
                processed,
                self.VESSEL_WINDOW['center'],
                self.VESSEL_WINDOW['width']
            )
            
            # Vessel enhancement
            if self.use_vessel_enhance and HAS_SKIMAGE:
                vessel = self.enhance_vessels(output)
                # Blend original with vessel-enhanced
                output = 0.7 * output + 0.3 * vessel
            
            # CLAHE
            if self.use_clahe:
                output = self.apply_clahe(output)
        
        return output
    
    def preprocess_volume(
        self, 
        volume: np.ndarray,
        return_channels: bool = False
    ) -> np.ndarray:
        """
        Preprocess entire 3D volume.
        
        Args:
            volume: 3D array (D, H, W) in HU values
            return_channels: If True, return (D, H, W, 3) multi-channel
            
        Returns:
            Preprocessed volume
        """
        processed_slices = []
        
        for i in range(volume.shape[0]):
            processed = self.preprocess_slice(
                volume[i], 
                return_channels=return_channels
            )
            processed_slices.append(processed)
        
        return np.stack(processed_slices, axis=0)


def create_visualization_image(
    image: np.ndarray,
    preprocessor: Optional[KaggleWinningPreprocessor] = None,
    add_annotations: bool = True
) -> np.ndarray:
    """
    Create a visualization image with preprocessing applied.
    
    Returns RGB image ready for display.
    
    Args:
        image: Input CT slice in HU values
        preprocessor: Preprocessor instance (creates default if None)
        add_annotations: Add bounding boxes for high-intensity regions
        
    Returns:
        RGB visualization image (H, W, 3), values 0-255
    """
    if preprocessor is None:
        preprocessor = KaggleWinningPreprocessor(
            use_multi_window=True,
            use_skull_strip=True,
            use_vessel_enhance=False,  # Skip for visualization speed
            use_clahe=True
        )
    
    # Get multi-window RGB
    rgb = preprocessor.create_multi_window_image(image)
    
    # Apply CLAHE to each channel
    if preprocessor.use_clahe:
        for i in range(3):
            rgb[:, :, i] = preprocessor.apply_clahe(rgb[:, :, i])
    
    # Convert to 0-255 range
    rgb = (rgb * 255).astype(np.uint8)
    
    # Add annotations for high-intensity regions
    if add_annotations and HAS_CV2:
        # Find high-intensity regions in vessel window
        vessel_window = preprocessor.apply_window(
            image, 
            preprocessor.VESSEL_WINDOW['center'],
            preprocessor.VESSEL_WINDOW['width']
        )
        
        # Threshold for bright regions
        binary = (vessel_window > 0.7).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw bounding boxes for significant regions
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 5000:  # Filter by size
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return rgb


# ============ PRESET CONFIGURATIONS ============

def get_kaggle_1st_place_preprocessor() -> KaggleWinningPreprocessor:
    """
    Get preprocessor matching Kaggle 1st place solution settings.
    
    Based on team "tomoon33" approach:
    - Multi-window visualization
    - Skull stripping
    - Vessel enhancement
    - CLAHE contrast enhancement
    """
    return KaggleWinningPreprocessor(
        use_multi_window=True,
        use_skull_strip=True,
        use_vessel_enhance=True,
        use_clahe=True,
        clahe_clip_limit=2.0,
        vessel_scales=(1, 2, 3, 4),
    )


def get_fast_preprocessor() -> KaggleWinningPreprocessor:
    """
    Get fast preprocessor for real-time applications.
    
    Disables slow operations like vessel enhancement.
    """
    return KaggleWinningPreprocessor(
        use_multi_window=True,
        use_skull_strip=True,
        use_vessel_enhance=False,
        use_clahe=True,
        clahe_clip_limit=2.0,
    )


def get_demo_preprocessor() -> KaggleWinningPreprocessor:
    """
    Get preprocessor for demo/visualization purposes.
    
    Optimized for visual output quality.
    """
    return KaggleWinningPreprocessor(
        use_multi_window=True,
        use_skull_strip=True,
        use_vessel_enhance=False,
        use_clahe=True,
        clahe_clip_limit=3.0,  # Higher for more dramatic contrast
    )


if __name__ == '__main__':
    print("=" * 60)
    print("Kaggle Winning Preprocessor - Feature Check")
    print("=" * 60)
    print(f"scikit-image available: {HAS_SKIMAGE}")
    print(f"OpenCV available: {HAS_CV2}")
    print()
    
    # Create test preprocessor
    preprocessor = get_kaggle_1st_place_preprocessor()
    print("Preprocessor settings:")
    print(f"  Multi-window: {preprocessor.use_multi_window}")
    print(f"  Skull strip: {preprocessor.use_skull_strip}")
    print(f"  Vessel enhance: {preprocessor.use_vessel_enhance}")
    print(f"  CLAHE: {preprocessor.use_clahe}")
    print()
    
    # Test with synthetic data
    test_image = np.random.randn(512, 512) * 100 + 40  # Simulate CT HU values
    print(f"Test image shape: {test_image.shape}")
    
    # Single channel processing
    result = preprocessor.preprocess_slice(test_image, return_channels=False)
    print(f"Single channel output shape: {result.shape}")
    print(f"Output range: [{result.min():.3f}, {result.max():.3f}]")
    
    # Multi-channel processing  
    result_rgb = preprocessor.preprocess_slice(test_image, return_channels=True)
    print(f"Multi-channel output shape: {result_rgb.shape}")
    
    print()
    print("✅ Preprocessor initialized successfully!")
