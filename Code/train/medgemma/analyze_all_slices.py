"""
MedGemma Multi-Slice Analyzer
Analyzes ALL DICOM slices in a folder/series to find potential aneurysms.

Usage:
    python analyze_all_slices.py <path_to_dicom_folder>
    python analyze_all_slices.py <series_uid>  (loads from ZIP)
"""

import sys
from pathlib import Path
import zipfile
import io
import numpy as np
from typing import List, Dict
import os

sys.path.insert(0, str(Path(__file__).parent))

from config import DATASET_ZIP, OUTPUT_DIR
from inference import MedGemmaInference


def load_all_slices_from_folder(folder_path: str) -> List[Dict]:
    """Load all DICOM files from a folder."""
    import pydicom
    
    folder = Path(folder_path)
    dcm_files = sorted(list(folder.glob("*.dcm")) + list(folder.glob("*.DCM")))
    
    if not dcm_files:
        raise ValueError(f"No DICOM files found in {folder_path}")
    
    print(f"📂 Found {len(dcm_files)} DICOM files in folder")
    
    slices = []
    for i, dcm_file in enumerate(dcm_files):
        dcm = pydicom.dcmread(str(dcm_file))
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Apply rescale
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        pixel_array = pixel_array * slope + intercept
        
        slices.append({
            'index': i,
            'filename': dcm_file.name,
            'pixel_array': pixel_array,
            'instance_number': getattr(dcm, 'InstanceNumber', i)
        })
    
    return slices


def load_all_slices_from_zip(series_uid: str) -> List[Dict]:
    """Load all DICOM slices for a series from the dataset ZIP."""
    import pydicom
    
    print(f"📂 Loading from: {DATASET_ZIP}")
    print(f"📋 Series UID: {series_uid[:50]}...")
    
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
        series_path = f"series/{series_uid}/"
        dcm_files = sorted([f for f in zf.namelist() 
                           if f.startswith(series_path) and not f.endswith('/')])
        
        if not dcm_files:
            raise ValueError(f"No files found for Series UID: {series_uid}")
        
        print(f"✅ Found {len(dcm_files)} DICOM slices")
        
        slices = []
        for i, dcm_file in enumerate(dcm_files):
            with zf.open(dcm_file) as f:
                dcm = pydicom.dcmread(io.BytesIO(f.read()))
                pixel_array = dcm.pixel_array.astype(np.float32)
                
                slope = float(getattr(dcm, 'RescaleSlope', 1))
                intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                pixel_array = pixel_array * slope + intercept
                
                slices.append({
                    'index': i,
                    'filename': Path(dcm_file).name,
                    'pixel_array': pixel_array,
                    'instance_number': getattr(dcm, 'InstanceNumber', i)
                })
    
    return slices


def analyze_all_slices(slices: List[Dict], sample_every: int = 10) -> Dict:
    """
    Analyze slices using MedGemma.
    
    Args:
        slices: List of slice data
        sample_every: Analyze every Nth slice (for speed). Set to 1 for ALL.
    """
    print(f"\n🧠 Initializing MedGemma...")
    model = MedGemmaInference()
    
    # Select slices to analyze
    if sample_every > 1:
        selected_indices = list(range(0, len(slices), sample_every))
        print(f"\n📊 Analyzing every {sample_every}th slice ({len(selected_indices)} of {len(slices)} total)")
    else:
        selected_indices = list(range(len(slices)))
        print(f"\n📊 Analyzing ALL {len(slices)} slices (this may take a while...)")
    
    # Short prompt for quick analysis
    quick_prompt = """Look at this brain CTA slice. 
    Answer ONLY: "ANEURYSM DETECTED" or "NO ANEURYSM DETECTED" 
    If detected, add location (e.g., "at left MCA").
    Be brief, max 20 words."""
    
    results = {
        'total_slices': len(slices),
        'analyzed_slices': len(selected_indices),
        'findings': [],
        'summary': None
    }
    
    print("\n🔬 Analyzing slices...")
    for count, idx in enumerate(selected_indices):
        slice_data = slices[idx]
        progress = f"[{count+1}/{len(selected_indices)}]"
        
        print(f"  {progress} Slice {idx+1}: {slice_data['filename'][:40]}...", end=" ")
        
        try:
            response = model.analyze_image(
                slice_data['pixel_array'],
                prompt=quick_prompt,
                system_prompt="You are a radiologist. Be very brief."
            )
            
            # Check if aneurysm detected
            response_lower = response.lower()
            has_aneurysm = "aneurysm detected" in response_lower and "no aneurysm" not in response_lower
            
            if has_aneurysm:
                print("⚠️ POTENTIAL FINDING!")
                results['findings'].append({
                    'slice_index': idx,
                    'slice_number': idx + 1,
                    'filename': slice_data['filename'],
                    'response': response.strip()
                })
            else:
                print("✓ Clear")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return results


def main():
    print("=" * 60)
    print("🏥 MEDGEMMA MULTI-SLICE ANALYZER")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_all_slices.py <path_to_dicom_folder>")
        print("  python analyze_all_slices.py <series_uid>")
        print("\nExample:")
        print("  python analyze_all_slices.py 1.2.826.0.1.3680043.8.498.10035643...")
        return
    
    input_path = sys.argv[1]
    sample_every = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    # Determine if it's a folder path or series UID
    if os.path.isdir(input_path):
        print(f"\n📁 Loading from folder: {input_path}")
        slices = load_all_slices_from_folder(input_path)
    else:
        print(f"\n📁 Loading Series UID from dataset ZIP")
        slices = load_all_slices_from_zip(input_path)
    
    # Analyze
    results = analyze_all_slices(slices, sample_every=sample_every)
    
    # Print results
    print("\n" + "=" * 60)
    print("📋 ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total slices: {results['total_slices']}")
    print(f"Slices analyzed: {results['analyzed_slices']}")
    print(f"Potential findings: {len(results['findings'])}")
    
    if results['findings']:
        print("\n⚠️ SLICES WITH POTENTIAL ANEURYSMS:")
        print("-" * 60)
        for finding in results['findings']:
            print(f"  📍 Slice {finding['slice_number']}: {finding['filename']}")
            print(f"     Response: {finding['response'][:100]}...")
            print()
    else:
        print("\n✅ No aneurysms detected in analyzed slices")
    
    # Save results
    output_file = OUTPUT_DIR / "multi_slice_report.txt"
    with open(output_file, 'w') as f:
        f.write("MEDGEMMA MULTI-SLICE ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total slices: {results['total_slices']}\n")
        f.write(f"Analyzed: {results['analyzed_slices']}\n")
        f.write(f"Findings: {len(results['findings'])}\n\n")
        
        if results['findings']:
            f.write("POTENTIAL ANEURYSM LOCATIONS:\n")
            f.write("-" * 40 + "\n")
            for finding in results['findings']:
                f.write(f"\nSlice {finding['slice_number']}: {finding['filename']}\n")
                f.write(f"Response: {finding['response']}\n")
    
    print(f"\n✅ Report saved to: {output_file}")
    print("\n💡 TIP: To analyze ALL slices (slower), run:")
    print(f"   python analyze_all_slices.py <uid> 1")


if __name__ == "__main__":
    main()
