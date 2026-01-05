"""
Step-by-Step Test Script for Aneurysm Detection Pipeline

This script shows you what happens at each stage:
1. Load data from ZIP
2. Preprocess DICOM images
3. Apply augmentations
4. Feed to model

Run from the ml/ folder:
    python test_pipeline.py
"""

import sys
import numpy as np
import pandas as pd
import zipfile
import io
from pathlib import Path

# Make sure we can import from data folder
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("STEP-BY-STEP PIPELINE TEST")
print("=" * 60)

# ============================================================
# STEP 1: Load train.csv from ZIP
# ============================================================
print("\n📊 STEP 1: Loading train.csv from ZIP file...")
print("-" * 40)

ZIP_PATH = "C:/Users/Rayan/Desktop/Main Project/rsna-intracranial-aneurysm-detection.zip"

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    print(f"✓ Opened ZIP file: {ZIP_PATH}")
    
    # List some files in the zip
    all_files = zf.namelist()
    print(f"✓ Total files in ZIP: {len(all_files)}")
    print(f"  First 5 files: {all_files[:5]}")
    
    # Load train.csv
    with zf.open('train.csv') as f:
        df = pd.read_csv(f)

print(f"\n✓ Loaded train.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\n  First 3 rows:")
print(df.head(3).to_string())

# Show label distribution
print(f"\n  Aneurysm cases: {df['Aneurysm Present'].sum()} / {len(df)}")

# ============================================================
# STEP 2: Load one DICOM series (preprocessing.py logic)
# ============================================================
print("\n\n🔬 STEP 2: Preprocessing - Load DICOM images...")
print("-" * 40)

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# Get first patient's series
sample_series = df.iloc[0]['SeriesInstanceUID']
sample_modality = df.iloc[0]['Modality']
print(f"Sample patient: {sample_series}")
print(f"Modality: {sample_modality}")

# Load DICOM files from this series
series_path = f"series/{sample_series}/"

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    # Find all DICOM files for this series
    dicom_files = [f for f in zf.namelist() if f.startswith(series_path) and not f.endswith('/')]
    print(f"✓ Found {len(dicom_files)} DICOM slices")
    
    # Load first 3 slices as example
    slices = []
    for i, dcm_name in enumerate(dicom_files[:3]):
        with zf.open(dcm_name) as f:
            dcm = pydicom.dcmread(io.BytesIO(f.read()))
            pixel_array = dcm.pixel_array.astype(np.float32)
            slices.append(pixel_array)
            
            if i == 0:
                print(f"\n  DICOM Metadata (first slice):")
                print(f"    Patient Age: {getattr(dcm, 'PatientAge', 'N/A')}")
                print(f"    Patient Sex: {getattr(dcm, 'PatientSex', 'N/A')}")
                print(f"    Slice Thickness: {getattr(dcm, 'SliceThickness', 'N/A')} mm")
                print(f"    Image Size: {pixel_array.shape}")
                print(f"    Pixel Range: [{pixel_array.min():.0f}, {pixel_array.max():.0f}]")

print(f"\n✓ Loaded {len(slices)} sample slices")

# ============================================================
# STEP 3: Apply CT Windowing (preprocessing.py logic)
# ============================================================
print("\n\n🪟 STEP 3: Apply CT Windowing...")
print("-" * 40)

# CT windowing parameters (from config.yaml)
WINDOW_CENTER = 40
WINDOW_WIDTH = 400

def apply_ct_window(image, center, width):
    """Apply CT windowing to make vessels visible."""
    lower = center - width / 2
    upper = center + width / 2
    
    # Clip to window range
    windowed = np.clip(image, lower, upper)
    
    # Normalize to [0, 1]
    windowed = (windowed - lower) / (upper - lower)
    
    return windowed

# Apply to first slice
original_slice = slices[0]
windowed_slice = apply_ct_window(original_slice, WINDOW_CENTER, WINDOW_WIDTH)

print(f"  Window: center={WINDOW_CENTER}, width={WINDOW_WIDTH}")
print(f"  Before windowing: [{original_slice.min():.0f}, {original_slice.max():.0f}]")
print(f"  After windowing:  [{windowed_slice.min():.3f}, {windowed_slice.max():.3f}]")
print(f"✓ Windowing makes brain vessels visible (removes bone brightness)")

# ============================================================
# STEP 4: Resize Volume (preprocessing.py logic)
# ============================================================
print("\n\n📐 STEP 4: Resize Volume...")
print("-" * 40)

from scipy.ndimage import zoom

# Target size (from config.yaml)
TARGET_SLICES = 32
TARGET_SIZE = (128, 128)

# Create a dummy volume for demonstration
dummy_volume = np.random.rand(50, 512, 512).astype(np.float32)
print(f"  Original volume shape: {dummy_volume.shape}")

# Calculate zoom factors
zoom_factors = (
    TARGET_SLICES / dummy_volume.shape[0],
    TARGET_SIZE[0] / dummy_volume.shape[1],
    TARGET_SIZE[1] / dummy_volume.shape[2]
)
print(f"  Zoom factors: {zoom_factors}")

# Resize (simplified - actual uses SimpleITK)
resized_volume = zoom(dummy_volume, zoom_factors, order=1)
print(f"  Resized volume shape: {resized_volume.shape}")
print(f"✓ Volume resized to {TARGET_SLICES} x {TARGET_SIZE[0]} x {TARGET_SIZE[1]}")

# ============================================================
# STEP 5: Apply Augmentations (augmentations.py logic)
# ============================================================
print("\n\n🔄 STEP 5: Apply Augmentations (Training Only)...")
print("-" * 40)

# Create sample volume
sample_volume = np.random.rand(32, 128, 128).astype(np.float32)

# Random flip
def random_flip(volume, axis, prob=0.5):
    if np.random.random() < prob:
        return np.flip(volume, axis=axis)
    return volume

# Random rotation
from scipy.ndimage import rotate as scipy_rotate

def random_rotate(volume, max_angle=15):
    angle = np.random.uniform(-max_angle, max_angle)
    return scipy_rotate(volume, angle, axes=(1, 2), reshape=False, order=1)

# Random intensity shift
def random_intensity(volume, shift_range=0.1):
    shift = np.random.uniform(-shift_range, shift_range)
    return np.clip(volume + shift, 0, 1)

# Normalize
def normalize(volume, mean=0.5, std=0.5):
    return (volume - mean) / std

print("  Augmentations applied:")

# Apply augmentations
aug_volume = sample_volume.copy()

# Flip
aug_volume = random_flip(aug_volume, axis=2)
print("    ✓ Random horizontal flip")

# Rotate
aug_volume = random_rotate(aug_volume, max_angle=15)
print("    ✓ Random rotation (±15°)")

# Intensity
aug_volume = random_intensity(aug_volume, shift_range=0.1)
print("    ✓ Random intensity shift")

# Normalize
aug_volume = normalize(aug_volume)
print("    ✓ Normalize to zero mean")

print(f"\n  Before augmentation: shape={sample_volume.shape}, range=[{sample_volume.min():.2f}, {sample_volume.max():.2f}]")
print(f"  After augmentation:  shape={aug_volume.shape}, range=[{aug_volume.min():.2f}, {aug_volume.max():.2f}]")
print("✓ Augmentations help prevent overfitting!")

# ============================================================
# STEP 6: Create PyTorch Tensor (dataset.py logic)
# ============================================================
print("\n\n🔧 STEP 6: Convert to PyTorch Tensor...")
print("-" * 40)

import torch

# Convert to tensor and add channel dimension
tensor_volume = torch.from_numpy(aug_volume.copy()).float()
tensor_volume = tensor_volume.unsqueeze(0)  # Add channel: (32,128,128) -> (1,32,128,128)

print(f"  NumPy shape: {aug_volume.shape}")
print(f"  Tensor shape: {tensor_volume.shape}  (C, D, H, W)")
print(f"  Tensor dtype: {tensor_volume.dtype}")

# Get labels for this patient
labels = df.iloc[0][['Left Infraclinoid Internal Carotid Artery',
                      'Right Infraclinoid Internal Carotid Artery',
                      'Aneurysm Present']].values.astype(np.float32)
label_tensor = torch.from_numpy(labels)

print(f"\n  Sample labels: {labels}")
print(f"  Label tensor: {label_tensor}")
print("✓ Ready for model input!")

# ============================================================
# SUMMARY
# ============================================================
print("\n\n" + "=" * 60)
print("✅ PIPELINE SUMMARY")
print("=" * 60)
print("""
1. dataset.py
   └── Opens ZIP, reads train.csv, manages data loading
   
2. preprocessing.py  
   └── Loads DICOM → CT Windowing → Resize to 32×128×128
   
3. augmentations.py (Training only)
   └── Flip → Rotate → Intensity → Normalize
   
4. Output: PyTorch tensor (1, 32, 128, 128) + labels (14,)
""")

print(f"\nTotal patients in dataset: {len(df)}")
print(f"Patients with aneurysm: {df['Aneurysm Present'].sum()}")
print(f"Patients without aneurysm: {len(df) - df['Aneurysm Present'].sum()}")

print("\n🎉 Pipeline test complete! Now you understand each step.")
