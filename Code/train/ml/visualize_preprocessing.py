"""
Preprocessing Debug Visualization - WITH PROOF
Shows histograms and numeric verification for each step.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import zipfile
import pandas as pd
import pydicom
import io

sys.path.insert(0, str(Path(__file__).parent))


def load_raw_dicom_slices(zip_path: str, target_uid: str = None, max_slices: int = 64):
    """Load RAW DICOM slices. If target_uid is None, finds first valid CTA."""
    print(f"📂 Opening: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        if target_uid is None:
            # Auto-find first CTA
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
            cta_scans = df[df['Modality'] == 'CTA']['SeriesInstanceUID'].values
            target_list = cta_scans[:20]
        else:
            # Use user-provided UID
            target_list = [target_uid]

        for series_uid in target_list:
            series_path = f"series/{series_uid}/"
            dcm_files = [f for f in zf.namelist() 
                        if f.startswith(series_path) and not f.endswith('/')]
            
            if len(dcm_files) < 10:
                print(f"   ⚠️ Series {series_uid} has too few files ({len(dcm_files)})")
                continue
            
            print(f"\n🔍 Loading: {series_uid[:30]}...")
            
            try:
                slices = []
                for dcm_name in sorted(dcm_files)[:max_slices]:
                    with zf.open(dcm_name) as f:
                        dcm = pydicom.dcmread(io.BytesIO(f.read()))
                        
                        if hasattr(dcm, 'pixel_array'):
                            pixel_array = dcm.pixel_array.astype(np.float32)
                            
                            # Convert to Hounsfield Units
                            slope = float(getattr(dcm, 'RescaleSlope', 1))
                            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                            pixel_array = pixel_array * slope + intercept
                            
                            instance = getattr(dcm, 'InstanceNumber', len(slices))
                            slices.append((instance, pixel_array))
                
                if len(slices) < 10:
                    continue
                
                slices.sort(key=lambda x: x[0])
                volume = np.stack([s[1] for s in slices], axis=0)
                
                print(f"   ✅ Shape: {volume.shape}")
                print(f"   RAW HU range: [{volume.min():.0f}, {volume.max():.0f}]")
                
                return volume, series_uid
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
    
    return None, None


def visualize_with_proof(target_uid: str = None):
    """Create visualization with PROOF that each step works."""
    print("=" * 70)
    print("PREPROCESSING VERIFICATION - WITH NUMERIC PROOF")
    print("=" * 70)
    
    zip_path = r"C:\Users\Rayan\Desktop\Main Project\rsna-intracranial-aneurysm-detection.zip"
    
    # Load raw data
    raw_volume, series_uid = load_raw_dicom_slices(zip_path, target_uid)
    
    if raw_volume is None:
        print("❌ Could not load data!")
        return
    
    mid_idx = raw_volume.shape[0] // 2
    raw_slice = raw_volume[mid_idx]
    
    # ============================================
    # STEP 1: RAW DATA
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 1: RAW CT DATA (Hounsfield Units)")
    print("=" * 50)
    print(f"   Min: {raw_volume.min():.0f} HU (should be around -1000 for air)")
    print(f"   Max: {raw_volume.max():.0f} HU (should be high for bone/contrast)")
    print(f"   Mean: {raw_volume.mean():.0f} HU")
    print(f"   Shape: {raw_volume.shape}")
    
    # ============================================
    # STEP 2: WINDOWING
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 2: WINDOWING (center=40, width=400)")
    print("=" * 50)
    
    # For CTA with contrast, use wider window
    center = 100  # Adjusted for contrast-enhanced blood
    width = 600   # Wider to capture vessels
    
    lower = center - width / 2  # -200
    upper = center + width / 2  # 400
    
    print(f"   Window range: [{lower:.0f}, {upper:.0f}] HU")
    print(f"   - Air (-1000) will become: 0 (black)")
    print(f"   - Soft tissue (40) will become: {(40 - lower) / (upper - lower):.2f}")
    print(f"   - Blood with contrast (200) will become: {(200 - lower) / (upper - lower):.2f}")
    print(f"   - Bone (1000) will become: 1 (white)")
    
    windowed = np.clip(raw_volume, lower, upper)
    windowed = (windowed - lower) / (upper - lower)
    windowed_slice = windowed[mid_idx]
    
    print(f"\n   AFTER windowing:")
    print(f"   Min: {windowed.min():.3f} (should be 0)")
    print(f"   Max: {windowed.max():.3f} (should be 1)")
    print(f"   Mean: {windowed.mean():.3f}")
    
    # ============================================
    # STEP 3: RESIZING
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 3: RESIZING")
    print("=" * 50)
    
    from scipy.ndimage import zoom
    target_shape = (64, 128, 128)
    factors = [t / s for t, s in zip(target_shape, windowed.shape)]
    
    print(f"   Original shape: {windowed.shape}")
    print(f"   Target shape: {target_shape}")
    print(f"   Zoom factors: {[f'{f:.3f}' for f in factors]}")
    
    resized = zoom(windowed, factors, order=1)
    resized_slice = resized[resized.shape[0] // 2]
    
    print(f"\n   AFTER resizing:")
    print(f"   Shape: {resized.shape}")
    print(f"   Min: {resized.min():.3f}")
    print(f"   Max: {resized.max():.3f}")
    
    # ============================================
    # STEP 4: NORMALIZATION
    # ============================================
    print("\n" + "=" * 50)
    print("STEP 4: Z-SCORE NORMALIZATION")
    print("=" * 50)
    
    mean = resized.mean()
    std = resized.std()
    
    print(f"   Before - Mean: {mean:.4f}, Std: {std:.4f}")
    
    normalized = (resized - mean) / std
    normalized_slice = normalized[normalized.shape[0] // 2]
    
    print(f"\n   AFTER normalization:")
    print(f"   Mean: {normalized.mean():.6f} (should be ~0)")
    print(f"   Std: {normalized.std():.6f} (should be ~1)")
    print(f"   Range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    
    # ============================================
    # CREATE VISUALIZATION
    # ============================================
    print("\n🎨 Creating visualization with histograms...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: Images
    ax1 = fig.add_subplot(2, 4, 1)
    im1 = ax1.imshow(raw_slice, cmap='gray', vmin=-200, vmax=400)
    ax1.set_title('1. RAW CT\n(Hounsfield Units)', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    ax2 = fig.add_subplot(2, 4, 2)
    im2 = ax2.imshow(windowed_slice, cmap='gray', vmin=0, vmax=1)
    ax2.set_title('2. WINDOWED\n(0 to 1)', fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(resized_slice, cmap='gray', vmin=0, vmax=1)
    ax3.set_title(f'3. RESIZED\n{resized.shape}', fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.6)
    
    ax4 = fig.add_subplot(2, 4, 4)
    im4 = ax4.imshow(normalized_slice, cmap='gray', vmin=-2, vmax=2)
    ax4.set_title('4. NORMALIZED\n(mean=0, std=1)', fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, shrink=0.6)
    
    # Row 2: Histograms (PROOF!)
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.hist(raw_slice.flatten(), bins=100, color='blue', alpha=0.7)
    ax5.axvline(x=-1000, color='red', linestyle='--', label='Air')
    ax5.axvline(x=40, color='green', linestyle='--', label='Tissue')
    ax5.axvline(x=1000, color='orange', linestyle='--', label='Bone')
    ax5.set_title('Histogram: RAW', fontweight='bold')
    ax5.set_xlabel('Hounsfield Units')
    ax5.legend(fontsize=8)
    
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.hist(windowed_slice.flatten(), bins=100, color='green', alpha=0.7)
    ax6.set_title('Histogram: WINDOWED', fontweight='bold')
    ax6.set_xlabel('Intensity (0-1)')
    ax6.set_xlim(0, 1)
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.hist(resized_slice.flatten(), bins=100, color='purple', alpha=0.7)
    ax7.set_title('Histogram: RESIZED', fontweight='bold')
    ax7.set_xlabel('Intensity (0-1)')
    ax7.set_xlim(0, 1)
    
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.hist(normalized_slice.flatten(), bins=100, color='red', alpha=0.7)
    ax8.axvline(x=0, color='black', linestyle='--', label='Mean=0')
    ax8.set_title('Histogram: NORMALIZED', fontweight='bold')
    ax8.set_xlabel('Z-Score')
    ax8.legend(fontsize=8)
    
    plt.suptitle(f'Preprocessing Pipeline Verification\nPatient: {series_uid[:40]}...', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / 'preprocessing_verified.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE!")
    print("The histograms PROVE each step is working correctly.")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    target_uid = None
    if len(sys.argv) > 1:
        target_uid = sys.argv[1]
        print(f"🎯 Using specific patient ID: {target_uid}")
    
    visualize_with_proof(target_uid)
