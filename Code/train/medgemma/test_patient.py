"""
Test MedGemma with a specific patient Series UID.
Usage: python test_patient.py <series_uid>
"""

import sys
from pathlib import Path
import zipfile
import io
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import DATASET_ZIP, OUTPUT_DIR
from inference import MedGemmaInference


def analyze_patient(series_uid: str):
    """Analyze a specific patient by Series UID."""
    import pydicom
    
    print("=" * 60)
    print("🏥 MEDGEMMA - Specific Patient Analysis")
    print("=" * 60)
    print(f"📋 Series UID: {series_uid[:50]}...")
    
    # Load patient from ZIP
    print(f"\n📂 Loading from: {DATASET_ZIP}")
    
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
        series_path = f"series/{series_uid}/"
        dcm_files = sorted([f for f in zf.namelist() 
                           if f.startswith(series_path) and not f.endswith('/')])
        
        if not dcm_files:
            print(f"❌ No files found for Series UID: {series_uid}")
            return
        
        print(f"✅ Found {len(dcm_files)} DICOM slices")
        
        # Load middle slice
        mid_idx = len(dcm_files) // 2
        mid_file = dcm_files[mid_idx]
        print(f"📄 Loading slice {mid_idx+1}/{len(dcm_files)}: {Path(mid_file).name}")
        
        with zf.open(mid_file) as f:
            dcm = pydicom.dcmread(io.BytesIO(f.read()))
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            pixel_array = pixel_array * slope + intercept
    
    print(f"✅ Loaded slice: shape={pixel_array.shape}")
    
    # Initialize MedGemma
    print("\n🧠 Initializing MedGemma...")
    model = MedGemmaInference()
    
    # Analyze
    print("\n🔬 Analyzing CT scan...")
    result = model.analyze_image(
        pixel_array,
        prompt="Analyze this brain CT angiography (CTA) slice for any signs of intracranial aneurysm. Describe what you see and provide your assessment."
    )
    
    # Display
    print("\n" + "=" * 60)
    print("📋 MEDGEMMA ANALYSIS REPORT")
    print("=" * 60)
    print(f"Series UID: {series_uid[:50]}...")
    print("-" * 60)
    print(result)
    print("=" * 60)
    
    # Save
    output_file = OUTPUT_DIR / f"report_{series_uid[:20]}.txt"
    with open(output_file, 'w') as f:
        f.write(f"Series UID: {series_uid}\n")
        f.write("-" * 40 + "\n")
        f.write(result)
    print(f"\n✅ Report saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        uid = sys.argv[1]
    else:
        # Default test UID
        uid = "1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381"
    
    analyze_patient(uid)
