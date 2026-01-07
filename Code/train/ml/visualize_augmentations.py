"""
Augmentation Visualization Script
Shows how we "fake" new data using 3D Augmentations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import zipfile
import pandas as pd
import pydicom
import io
from scipy.ndimage import rotate

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Import our augmentation classes
from data.augmentations import RandomRotate3D, GaussianNoise, RandomFlip3D


def load_real_sample_base(zip_path: str, target_uid: str = None):
    """Load a sample and preprocess it to a clean state (0-1 range)."""
    print(f"📂 Opening: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if target_uid is None:
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
            cta_scans = df[df['Modality'] == 'CTA']['SeriesInstanceUID'].values
            target_list = cta_scans[:20]
        else:
            target_list = [target_uid]

        for series_uid in target_list:
            series_path = f"series/{series_uid}/"
            dcm_files = [f for f in zf.namelist() 
                        if f.startswith(series_path) and not f.endswith('/')]
            
            if len(dcm_files) < 10:
                continue
            
            print(f"\n🔍 Loading: {series_uid[:30]}...")
            
            try:
                # Load RAW
                slices = []
                for dcm_name in sorted(dcm_files)[:64]: # Limit Z for speed
                    with zf.open(dcm_name) as f:
                        dcm = pydicom.dcmread(io.BytesIO(f.read()))
                        if hasattr(dcm, 'pixel_array'):
                            pixel_array = dcm.pixel_array.astype(np.float32)
                            slope = float(getattr(dcm, 'RescaleSlope', 1))
                            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                            pixel_array = pixel_array * slope + intercept
                            slices.append((getattr(dcm, 'InstanceNumber', 0), pixel_array))
                
                if len(slices) < 10: continue
                slices.sort(key=lambda x: x[0])
                volume = np.stack([s[1] for s in slices], axis=0)
                
                # WINDOWING (Center=100, Width=600 for vessels)
                center, width = 100, 600
                low, high = center - width/2, center + width/2
                volume = np.clip(volume, low, high)
                volume = (volume - low) / (high - low)
                
                # RESIZING (Simple zoom for demo)
                from scipy.ndimage import zoom
                target = (64, 128, 128)
                factors = [t/s for t,s in zip(target, volume.shape)]
                volume = zoom(volume, factors, order=1)
                
                return volume, series_uid
                
            except Exception as e:
                print(e)
                continue
    return None, None


def visualize_augmentations(target_uid: str = None):
    print("=" * 60)
    print("AUGMENTATION DEMO")
    print("=" * 60)
    
    zip_path = r"C:\Users\Rayan\Desktop\Main Project\rsna-intracranial-aneurysm-detection.zip"
    
    # 1. Get Base Volume
    volume, series_uid = load_real_sample_base(zip_path, target_uid)
    if volume is None: return
    
    mid_slice = volume.shape[0] // 2
    
    # Define Augmentations
    # Force them to happen (prob=1.0)
    rotator = RandomRotate3D(angle_range=20, axes=(1, 2)) # xy rotation
    noiser = GaussianNoise(std=0.05, prob=1.0)
    flipper = RandomFlip3D(flip_prob=1.0, axes=(2,)) # horizontal flip
    
    # Apply them ONE BY ONE to the base volume
    # Note: Our classes expect (D, H, W)
    
    # Rotation
    # We cheat to force a visible rotation for the demo
    rotated_vol = rotate(volume, angle=30, axes=(1, 2), reshape=False, mode='constant', cval=0)
    
    # Noise
    noisy_vol = noiser(volume.copy())
    
    # Flip
    flipped_vol = np.flip(volume.copy(), axis=2) # Flip width
    
    # Visualization
    print("\n🎨 Creating comparison image...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    # 1. Original
    axes[0].imshow(volume[mid_slice], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('1. Original Data\n(Clean)', fontsize=12, fontweight='bold')
    
    # 2. Rotated
    axes[1].imshow(rotated_vol[mid_slice], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('2. Random Rotation\n(Simulates tilted head)', fontsize=12, fontweight='bold')
    
    # 3. Noisy
    axes[2].imshow(noisy_vol[mid_slice], cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('3. Gaussian Noise\n(Simulates bad scanner)', fontsize=12, fontweight='bold')
    
    # 4. Flipped
    axes[3].imshow(flipped_vol[mid_slice], cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('4. Horizontal Flip\n(Left <-> Right)', fontsize=12, fontweight='bold')
    
    for ax in axes: ax.axis('off')
    
    plt.suptitle(f"Data Augmentation Pipeline\nPatient: {series_uid[:30]}...", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    out_path = Path(__file__).parent / 'augmentation_demo.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    import sys
    uid = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_augmentations(uid)
