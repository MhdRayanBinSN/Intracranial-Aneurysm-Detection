"""
MedGemma Demo Script
Analyzes a sample CT scan from your dataset.
"""

import sys
from pathlib import Path
import zipfile
import io
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASET_ZIP, OUTPUT_DIR
from inference import MedGemmaInference


def load_sample_ct_from_zip(zip_path: str = DATASET_ZIP) -> tuple:
    """Load a sample CT slice from the dataset ZIP."""
    import pydicom
    import pandas as pd
    
    print(f"📂 Loading sample from: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get a CTA scan
        with zf.open('train.csv') as f:
            df = pd.read_csv(f)
        
        # Get first CTA
        uid = df[df['Modality'] == 'CTA']['SeriesInstanceUID'].values[0]
        print(f"🔍 Selected Patient: {uid[:30]}...")
        
        # Get files
        series_path = f"series/{uid}/"
        dcm_files = [f for f in zf.namelist() 
                    if f.startswith(series_path) and not f.endswith('/')]
        
        if not dcm_files:
            raise ValueError("No DICOM files found!")
        
        # Load middle slice
        mid_file = sorted(dcm_files)[len(dcm_files) // 2]
        print(f"📄 Loading slice: {Path(mid_file).name}")
        
        with zf.open(mid_file) as f:
            dcm = pydicom.dcmread(io.BytesIO(f.read()))
            pixel_array = dcm.pixel_array.astype(np.float32)
            
            # Apply rescale
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            pixel_array = pixel_array * slope + intercept
        
        return pixel_array, uid


def main():
    print("=" * 60)
    print("🏥 MEDGEMMA DEMO - Aneurysm Detection")
    print("=" * 60)
    
    # Load sample CT
    try:
        ct_slice, patient_id = load_sample_ct_from_zip()
        print(f"✅ Loaded CT slice: shape={ct_slice.shape}")
    except Exception as e:
        print(f"❌ Error loading CT: {e}")
        print("\n💡 Using a test image instead...")
        # Create a simple test image
        from PIL import Image
        ct_slice = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        patient_id = "test_patient"
    
    # Initialize MedGemma
    print("\n🧠 Initializing MedGemma...")
    try:
        model = MedGemmaInference()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n📋 Troubleshooting:")
        print("1. Make sure you set HF_TOKEN:")
        print("   set HF_TOKEN=your_huggingface_token")
        print("\n2. Get token from: https://huggingface.co/settings/tokens")
        print("\n3. Accept MedGemma license at:")
        print("   https://huggingface.co/google/medgemma-4b-it")
        return
    
    # Analyze
    print("\n🔬 Analyzing CT scan...")
    result = model.analyze_image(
        ct_slice,
        prompt="Analyze this brain CT angiography (CTA) slice for any signs of intracranial aneurysm. Describe what you see and provide your assessment."
    )
    
    # Display result
    print("\n" + "=" * 60)
    print("📋 MEDGEMMA ANALYSIS REPORT")
    print("=" * 60)
    print(f"Patient ID: {patient_id[:30]}...")
    print("-" * 60)
    print(result)
    print("=" * 60)
    
    # Save result
    output_file = OUTPUT_DIR / "analysis_report.txt"
    with open(output_file, 'w') as f:
        f.write(f"Patient ID: {patient_id}\n")
        f.write("-" * 40 + "\n")
        f.write(result)
    print(f"\n✅ Report saved to: {output_file}")


if __name__ == "__main__":
    main()
