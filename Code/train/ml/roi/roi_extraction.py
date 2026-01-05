"""
ROI Extraction (1st Place Solution - Phase 3)

Implements coarse-to-fine ROI extraction:
1. Coarse model finds candidate region
2. DBSCAN clustering removes false positives
3. Fine models segment detailed vessels within ROI
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from scipy.ndimage import label, center_of_mass
from sklearn.cluster import DBSCAN


class ROIExtractor:
    """
    Extract Region of Interest using coarse-to-fine approach.
    
    Pipeline:
    1. Run coarse segmentation
    2. Binarize and cluster
    3. Find largest cluster centroid
    4. Crop fixed-size ROI
    """
    
    def __init__(
        self,
        roi_size_mm: Tuple[float, float, float] = (140, 140, 140),
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        Initialize ROI extractor.
        
        Args:
            roi_size_mm: ROI size in millimeters (D, H, W)
            spacing: Voxel spacing in mm
        """
        self.roi_size_mm = roi_size_mm
        self.spacing = spacing
        
        # Calculate ROI size in voxels
        self.roi_size_voxels = tuple(
            int(s / sp) for s, sp in zip(roi_size_mm, spacing)
        )
    
    def extract_roi_from_coarse_mask(
        self,
        coarse_mask: np.ndarray,
        volume: np.ndarray,
        min_cluster_size: int = 100,
    ) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
        """
        Extract ROI based on coarse segmentation mask.
        
        Args:
            coarse_mask: Binary mask from coarse model (D, H, W)
            volume: Full volume (D, H, W)
            min_cluster_size: Minimum voxels for valid cluster
            
        Returns:
            roi_volume: Cropped ROI volume
            slices: Tuple of slices used for cropping
        """
        # Step 1: Binarize mask
        binary_mask = (coarse_mask > 0.5).astype(np.uint8)
        
        # Step 2: Find connected components
        labeled, num_labels = label(binary_mask)
        
        if num_labels == 0:
            # No foreground found, return center ROI
            return self._extract_center_roi(volume)
        
        # Step 3: DBSCAN clustering to remove scattered false positives
        foreground_coords = np.argwhere(binary_mask > 0)
        
        if len(foreground_coords) < min_cluster_size:
            return self._extract_center_roi(volume)
        
        # Use DBSCAN
        clustering = DBSCAN(eps=5, min_samples=10).fit(foreground_coords)
        cluster_labels = clustering.labels_
        
        # Find largest cluster (excluding noise label -1)
        unique_labels = [l for l in np.unique(cluster_labels) if l >= 0]
        
        if not unique_labels:
            return self._extract_center_roi(volume)
        
        largest_cluster = max(
            unique_labels, 
            key=lambda l: (cluster_labels == l).sum()
        )
        
        # Step 4: Get centroid of largest cluster
        cluster_mask = cluster_labels == largest_cluster
        cluster_coords = foreground_coords[cluster_mask]
        centroid = cluster_coords.mean(axis=0).astype(int)
        
        # Step 5: Crop fixed-size ROI around centroid
        return self._extract_roi_around_point(volume, centroid)
    
    def _extract_center_roi(
        self, volume: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
        """Extract ROI from center of volume."""
        center = np.array(volume.shape) // 2
        return self._extract_roi_around_point(volume, center)
    
    def _extract_roi_around_point(
        self,
        volume: np.ndarray,
        center: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
        """
        Extract fixed-size ROI centered at a point.
        
        Args:
            volume: Full volume (D, H, W)
            center: Center point (d, h, w)
            
        Returns:
            roi_volume: Cropped volume
            slices: Tuple of slices
        """
        slices = []
        
        for dim, (c, roi_size, vol_size) in enumerate(zip(
            center, self.roi_size_voxels, volume.shape
        )):
            half_size = roi_size // 2
            
            # Calculate start and end
            start = max(0, c - half_size)
            end = min(vol_size, c + roi_size - half_size)
            
            # Adjust if ROI goes out of bounds
            if end - start < roi_size:
                if start == 0:
                    end = min(vol_size, roi_size)
                elif end == vol_size:
                    start = max(0, vol_size - roi_size)
            
            slices.append(slice(int(start), int(end)))
        
        roi_volume = volume[slices[0], slices[1], slices[2]]
        
        return roi_volume, tuple(slices)
    
    def compute_tight_bbox(
        self,
        mask: np.ndarray,
        margin_mm: float = 10.0,
    ) -> Tuple[slice, slice, slice]:
        """
        Compute tight bounding box around vessel mask with margins.
        
        Args:
            mask: Binary vessel mask
            margin_mm: Margin in millimeters
            
        Returns:
            Tuple of slices for bounding box
        """
        margin_voxels = tuple(int(margin_mm / s) for s in self.spacing)
        
        # Find non-zero coordinates
        coords = np.argwhere(mask > 0)
        
        if len(coords) == 0:
            # Empty mask, return full volume
            return (slice(None), slice(None), slice(None))
        
        # Get min/max for each dimension
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        
        slices = []
        for dim, (mn, mx, margin, shape) in enumerate(zip(
            mins, maxs, margin_voxels, mask.shape
        )):
            start = max(0, mn - margin)
            end = min(shape, mx + margin + 1)
            slices.append(slice(int(start), int(end)))
        
        return tuple(slices)


class VesselMaskedPooling:
    """
    Extract features for each anatomical location using vessel masks.
    
    This is the key technique from 1st place solution:
    - Use vessel masks to extract location-specific features
    - Each of 13 locations gets its own feature vector
    """
    
    # Mapping from mask label to anatomical location
    LOCATION_MAPPING = {
        1: "Left Infraclinoid ICA",
        2: "Right Infraclinoid ICA",
        3: "Left Supraclinoid ICA",
        4: "Right Supraclinoid ICA",
        5: "Left MCA",
        6: "Right MCA",
        7: "AComm",
        8: "Left ACA",
        9: "Right ACA",
        10: "Left PComm",
        11: "Right PComm", 
        12: "Basilar Tip",
        13: "Other Posterior",
    }
    
    def __init__(self, num_locations: int = 13):
        """
        Initialize masked pooling.
        
        Args:
            num_locations: Number of anatomical locations
        """
        self.num_locations = num_locations
    
    def pool_features_by_location(
        self,
        feature_map: np.ndarray,
        vessel_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Pool features for each anatomical location.
        
        Args:
            feature_map: Feature map from encoder (C, D, H, W)
            vessel_mask: Vessel segmentation with location labels (D, H, W)
            
        Returns:
            location_features: Features per location (num_locations, C)
        """
        num_channels = feature_map.shape[0]
        location_features = np.zeros((self.num_locations, num_channels))
        
        for loc in range(1, self.num_locations + 1):
            # Create mask for this location
            loc_mask = (vessel_mask == loc)
            
            if loc_mask.sum() == 0:
                # No voxels for this location, use zeros
                continue
            
            # Resize mask to match feature map if needed
            if loc_mask.shape != feature_map.shape[1:]:
                from scipy.ndimage import zoom
                scale = tuple(
                    f / m for f, m in zip(feature_map.shape[1:], loc_mask.shape)
                )
                loc_mask = zoom(loc_mask.astype(float), scale, order=0) > 0.5
            
            # Masked average pooling
            for c in range(num_channels):
                masked_features = feature_map[c][loc_mask]
                if len(masked_features) > 0:
                    location_features[loc - 1, c] = masked_features.mean()
        
        return location_features
    
    def pool_global_vessel_features(
        self,
        feature_map: np.ndarray,
        vessel_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Pool features over all vessels (for "Aneurysm Present" prediction).
        
        Args:
            feature_map: Feature map (C, D, H, W)
            vessel_mask: Binary vessel mask (D, H, W)
            
        Returns:
            global_features: Pooled features (C,)
        """
        # Union of all vessel masks
        binary_mask = (vessel_mask > 0)
        
        if binary_mask.sum() == 0:
            return feature_map.mean(axis=(1, 2, 3))
        
        # Resize if needed
        if binary_mask.shape != feature_map.shape[1:]:
            from scipy.ndimage import zoom
            scale = tuple(
                f / m for f, m in zip(feature_map.shape[1:], binary_mask.shape)
            )
            binary_mask = zoom(binary_mask.astype(float), scale, order=0) > 0.5
        
        # Masked pooling
        global_features = np.zeros(feature_map.shape[0])
        for c in range(feature_map.shape[0]):
            masked_features = feature_map[c][binary_mask]
            if len(masked_features) > 0:
                global_features[c] = masked_features.mean()
        
        return global_features


if __name__ == '__main__':
    print("=" * 60)
    print("ROI EXTRACTION TEST")
    print("=" * 60)
    
    # Test ROI extraction
    extractor = ROIExtractor(
        roi_size_mm=(140, 140, 140),
        spacing=(1.0, 1.0, 1.0),
    )
    
    # Create dummy volume and mask
    volume = np.random.randn(200, 256, 256).astype(np.float32)
    coarse_mask = np.zeros((200, 256, 256))
    coarse_mask[80:120, 100:156, 100:156] = 1  # Simulated vessel region
    
    # Extract ROI
    roi, slices = extractor.extract_roi_from_coarse_mask(coarse_mask, volume)
    
    print(f"Original volume shape: {volume.shape}")
    print(f"ROI shape: {roi.shape}")
    print(f"Slices used: {slices}")
    
    # Test vessel masked pooling
    print("\nTesting vessel-masked pooling...")
    pooler = VesselMaskedPooling(num_locations=13)
    
    # Dummy feature map and mask
    feature_map = np.random.randn(64, 32, 64, 64).astype(np.float32)
    vessel_mask = np.zeros((128, 256, 256), dtype=np.uint8)
    vessel_mask[50:70, 100:150, 100:150] = 5  # MCA region
    vessel_mask[60:80, 120:130, 120:130] = 12  # Basilar region
    
    location_features = pooler.pool_features_by_location(feature_map, vessel_mask)
    print(f"Location features shape: {location_features.shape}")
    print(f"Non-zero locations: {(location_features.sum(axis=1) != 0).sum()}")
    
    global_features = pooler.pool_global_vessel_features(feature_map, vessel_mask)
    print(f"Global features shape: {global_features.shape}")
    
    print("\n✅ ROI extraction module working!")
