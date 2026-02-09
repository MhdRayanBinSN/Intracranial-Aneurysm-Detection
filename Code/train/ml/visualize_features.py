"""
Feature Visualizer (The Barcode)
Shows the 512 "Hidden Features" extracted by the ResNet3D.
"""

import torch
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

from models.model import AneurysmDetector
from data.preprocessing import DicomPreprocessor

def load_random_patient_volume():
    """Load a real patient volume."""
    zip_path = r"C:\Users\Rayan\Desktop\Main Project\rsna-intracranial-aneurysm-detection.zip"
    print(f"📂 Loading data from: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('train.csv') as f:
            df = pd.read_csv(f)
        
        # Get a CTA scan
        uid = df[df['Modality'] == 'CTA']['SeriesInstanceUID'].values[0]
        print(f"🔍 Analyzing Patient: {uid[:30]}...")
        
        # Load slices (Simplified version of what we did before)
        # In a real app we'd use the full preprocessor, but here we just want a quick sample
        series_path = f"series/{uid}/"
        dcm_files = [f for f in zf.namelist() if f.startswith(series_path) and not f.endswith('/')]
        
        slices = []
        for dcm_name in sorted(dcm_files)[:64]: # Limit to 64 slices
            with zf.open(dcm_name) as f:
                dcm = pydicom.dcmread(io.BytesIO(f.read()))
                if hasattr(dcm, 'pixel_array'):
                    arr = dcm.pixel_array.astype(np.float32)
                    slices.append(arr)
        
        if not slices:
            print("❌ No slices found.")
            return None
            
        # Stack
        volume = np.stack(slices, axis=0) # (D, H, W)
        
        # Resize to 128x128x64 (Model input size)
        from scipy.ndimage import zoom
        target_shape = (64, 128, 128)
        factors = [t/s for t,s in zip(target_shape, volume.shape)]
        volume = zoom(volume, factors, order=1)
        
        # Normalize
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)
        
        # Add Batch & Channel dims: (1, 1, 64, 128, 128)
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
        return tensor

def visualize_features():
    print("=" * 60)
    print("🧠 BRAIN FEATURE EXTRACTOR (RESNET-18)")
    print("=" * 60)
    
    # 1. Load Data
    x = load_random_patient_volume()
    if x is None: return

    # 2. Load Model
    print("\nINIT MODEL: ResNet3D-18")
    model = AneurysmDetector(backbone_name='resnet3d_18')
    model.eval()
    
    # 3. Extract Features
    print("⚡ Running Forward Pass...")
    with torch.no_grad():
        # This returns the dict with 'features'
        output = model(x)
        features = output['features'][0] # Take first item in batch
    
    print(f"✅ Extraction Complete!")
    print(f"   Shape: {features.shape} (It is a vector of 512 numbers)")
    print(f"   Values: {features[:5].numpy()} ... (and 507 more)")
    
    # 4. Visualize as Barcode
    feature_vals = features.numpy()
    
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Reshape to 16x32 for a nicer grid view, or 1x512 for barcode
    # Let's do a barcode style (1, 512)
    im = ax.imshow(feature_vals.reshape(1, -1), aspect='auto', cmap='viridis')
    
    ax.set_yticks([])
    ax.set_title(f"THE BRAIN BARCODE\n(These are the 512 Features - Unique to this Patient)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature Index (0 to 511)")
    
    plt.colorbar(im, label='Activation Strength')
    plt.tight_layout()
    
    out_path = Path(__file__).parent / 'feature_barcode.png'
    plt.savefig(out_path, dpi=150, facecolor='white')
    print(f"\n✅ Saved Barcode to: {out_path}")
    plt.show()

if __name__ == "__main__":
    visualize_features()
