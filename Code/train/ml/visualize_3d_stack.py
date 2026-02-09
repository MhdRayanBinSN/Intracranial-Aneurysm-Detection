"""
3D Stack Visualization
Shows how the 3D volume is just a stack of 2D slices.
Generates a montage of slices from bottom to top.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import zipfile
import pandas as pd
import pydicom
import io

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def load_and_visualize_stack(target_uid: str = None):
    print("=" * 60)
    print("3D STACK VISUALIZER")
    print("=" * 60)
    
    zip_path = r"C:\Users\Rayan\Desktop\Main Project\rsna-intracranial-aneurysm-detection.zip"
    
    print(f"📂 Opening ZIP...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find a scan
        if target_uid is None:
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
            # Pick a random one for variety, or the first one
            target_uid = df[df['Modality'] == 'CTA']['SeriesInstanceUID'].values[0]
            
        print(f"🔍 Loading Series: {target_uid[:30]}...")
        
        series_path = f"series/{target_uid}/"
        dcm_files = [f for f in zf.namelist() if f.startswith(series_path) and not f.endswith('/')]
        
        # Load all slices
        slices = []
        for dcm_name in sorted(dcm_files):
            with zf.open(dcm_name) as f:
                dcm = pydicom.dcmread(io.BytesIO(f.read()))
                if hasattr(dcm, 'pixel_array'):
                    # Basic preprocessing for visibility
                    arr = dcm.pixel_array.astype(np.float32)
                    # Windowing (bone/tissue)
                    arr = np.clip(arr, -100, 300) 
                    slices.append((getattr(dcm, 'InstanceNumber', 0), arr))
        
        slices.sort(key=lambda x: x[0])
        volume = np.stack([s[1] for s in slices], axis=0) # THE 3D STACKING
        
        print(f"✅ 3D Volume Constructed!")
        print(f"   Shape: {volume.shape} (Depth, Height, Width)")
        print(f"   We have {volume.shape[0]} slices stacked on top of each other.")
        
        # Create Montage (Grid of Slices)
        # We'll take 16 slices evenly spaced
        indices = np.linspace(0, volume.shape[0]-1, 16, dtype=int)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f"3D Stack Visualization\n(Scanning from Bottom to Top of Head)", fontsize=16, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            slice_idx = indices[i]
            ax.imshow(volume[slice_idx], cmap='gray')
            ax.set_title(f"Slice {slice_idx}", fontsize=10)
            ax.axis('off')
            
        plt.tight_layout()
        
        out_path = Path(__file__).parent / '3d_stack_visualization.png'
        plt.savefig(out_path, dpi=150, facecolor='white')
        print(f"\n✅ Saved Montage to: {out_path}")
        plt.show()

if __name__ == "__main__":
    load_and_visualize_stack()
