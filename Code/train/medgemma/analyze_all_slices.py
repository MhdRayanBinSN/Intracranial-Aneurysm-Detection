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
import re


def parse_aneurysm_response(response: str) -> bool:
    """
    Parse MedGemma's structured response to determine if aneurysm detected.
    
    Handles multiple response formats:
    - "ANEURYSM: YES" / "ANEURYSM: NO"
    - "aneurysm detected" / "no aneurysm detected"
    - "YES" at start of line
    - Presence of anatomical location descriptions
    
    Returns True if aneurysm is likely detected.
    """
    response_lower = response.lower().strip()
    
    # Pattern 1: Structured fields used by config.py prompts.
    # Accept both legacy "ANEURYSM: YES" and current "ANEURYSM_DETECTED: YES".
    aneurysm_match = re.search(
        r'aneurysm(?:[_\s-]*detected)?\s*:\s*(yes|no)\b',
        response_lower,
    )
    if aneurysm_match:
        return aneurysm_match.group(1) == 'yes'
    
    # Pattern 2: Check for explicit NO statements first (to avoid false positives)
    negative_patterns = [
        r'aneurysm[_\s-]*detected\s*:\s*no',
        r'no aneurysm',
        r'aneurysm[:\s]+no',
        r'not\s+(see|detect|observe|identify)',
        r'no\s+(visible|apparent|obvious|clear)\s+aneurysm',
        r'negative for aneurysm',
        r'no evidence of',
        r'unremarkable',
        r'within normal limits',
    ]
    for pattern in negative_patterns:
        if re.search(pattern, response_lower):
            return False
    
    # Pattern 3: Check for positive indicators
    positive_patterns = [
        r'aneurysm[_\s-]*detected\s*:\s*yes',
        r'aneurysm\s+detected',
        r'aneurysm\s+identified',
        r'aneurysm\s+present',
        r'aneurysm\s+seen',
        r'aneurysm\s+noted',
        r'saccular\s+aneurysm',
        r'berry\s+aneurysm',
        r'outpouching\s+(from|at|of)',
        r'(ica|mca|aca|pcomm|acomm|basilar)\s+aneurysm',
        r'aneurysm.{0,30}(ica|mca|aca|pcomm|acomm|basilar)',
        r'suspicious\s+(for|of)\s+aneurysm',
        r'likely\s+aneurysm',
        r'compatible\s+with\s+aneurysm',
    ]
    for pattern in positive_patterns:
        if re.search(pattern, response_lower):
            return True
    
    # Pattern 4: Simple "YES" response with confidence
    if re.search(r'^yes\b', response_lower) or re.search(r'\byes\b.*confidence', response_lower):
        return True
    
    # Default: no detection
    return False


def extract_location_from_response(response: str) -> str:
    """Extract the anatomical location from MedGemma's response."""
    response_lower = response.lower()
    
    # Location patterns with priority, normalized to RSNA column names where possible.
    location_patterns = [
        (r'(left|right)\s+(infraclinoid|supraclinoid)\s+(?:internal\s+carotid\s+artery|ica)', lambda m: f"{m.group(1).title()} {m.group(2).title()} Internal Carotid Artery"),
        (r'(left|right)\s+(?:middle\s+cerebral\s+artery|mca)', lambda m: f"{m.group(1).title()} Middle Cerebral Artery"),
        (r'(left|right)\s+(?:anterior\s+cerebral\s+artery|aca)', lambda m: f"{m.group(1).title()} Anterior Cerebral Artery"),
        (r'(left|right)\s+(?:posterior\s+communicating\s+artery|pcomm|p-comm)', lambda m: f"{m.group(1).title()} Posterior Communicating Artery"),
        (r'acomm|a-comm|anterior\s+communicating', lambda m: "Anterior Communicating Artery"),
        (r'basilar\s+(tip|apex)', lambda m: "Basilar Tip"),
        (r'other\s+posterior\s+circulation|posterior\s+circulation', lambda m: "Other Posterior Circulation"),
        (r'(left|right)\s+(?:internal\s+carotid\s+artery|ica)', lambda m: f"{m.group(1).title()} Internal Carotid Artery"),
    ]
    
    for pattern, formatter in location_patterns:
        match = re.search(pattern, response_lower)
        if match:
            return formatter(match)
    
    # Check for LOCATION: field
    loc_match = re.search(r'location\s*:\s*([^\n]+)', response_lower)
    if loc_match and loc_match.group(1).strip() != 'n/a':
        return loc_match.group(1).strip().title()
    
    return "Unknown Location"


def extract_confidence_from_response(response: str) -> str:
    """Extract confidence level from MedGemma's response."""
    response_lower = response.lower()
    
    conf_match = re.search(r'confidence\s*:\s*(high|medium|low)', response_lower)
    if conf_match:
        return conf_match.group(1).upper()
    
    # Infer confidence from language
    if any(word in response_lower for word in ['definite', 'clear', 'obvious', 'certain']):
        return 'HIGH'
    if any(word in response_lower for word in ['possible', 'suspected', 'likely', 'probable']):
        return 'MEDIUM'
    if any(word in response_lower for word in ['uncertain', 'subtle', 'may be', 'questionable']):
        return 'LOW'
    
    return 'MEDIUM'


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


def analyze_with_context(
    slices: List[Dict], 
    model: MedGemmaInference,
    center_idx: int,
    modality: str = "CTA"
) -> Dict:
    """
    Analyze a slice using 3-slice context window for improved accuracy.
    
    Shows the model slices N-1, N, N+1 to provide 3D spatial context.
    A true aneurysm appears across multiple consecutive slices.
    
    Args:
        slices: All slices in the series
        model: MedGemma inference instance
        center_idx: Index of the center slice to analyze
        modality: Imaging modality
    
    Returns:
        dict with analysis results
    """
    n = len(slices)
    
    # Get adjacent slices
    prev_idx = max(0, center_idx - 1)
    next_idx = min(n - 1, center_idx + 1)
    
    # Extract pixel arrays
    pixel_arrays = [s['pixel_array'] for s in slices]
    
    # Use the model's context window analysis
    slice_info = f"Slice {center_idx + 1} of {n}"
    
    response = model.analyze_context_window(
        pixel_arrays=pixel_arrays,
        center_idx=center_idx,
        modality=modality,
        slice_info=slice_info,
    )
    
    return {
        'center_idx': center_idx,
        'context_indices': [prev_idx, center_idx, next_idx],
        'response': response,
    }


def analyze_all_slices(slices: List[Dict], sample_every: int = 10, use_context: bool = False) -> Dict:
    """
    Analyze slices using MedGemma.
    
    Args:
        slices: List of slice data
        sample_every: Analyze every Nth slice (for speed). Set to 1 for ALL.
        use_context: If True, use 3-slice context window for better 3D analysis.
    """
    print(f"\n🧠 Initializing MedGemma...")
    model = MedGemmaInference()
    
    if use_context:
        print("📐 Using 3-slice context window for better accuracy")
    
    # Select slices to analyze
    if sample_every > 1:
        selected_indices = list(range(0, len(slices), sample_every))
        print(f"\n📊 Analyzing every {sample_every}th slice ({len(selected_indices)} of {len(slices)} total)")
    else:
        selected_indices = list(range(len(slices)))
        print(f"\n📊 Analyzing ALL {len(slices)} slices (this may take a while...)")
    
    # Improved prompt for aneurysm detection (structured output)
    detection_prompt = """Analyze this brain CTA slice for intracranial aneurysms.

Look carefully for:
1. Saccular (berry) aneurysms: Round/lobulated outpouchings at vessel bifurcations
2. Bright contrast-filled bulges projecting from arteries
3. Common locations: ICA, MCA bifurcation, AComm, PComm, Basilar tip

Output EXACTLY in this format:
ANEURYSM: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
LOCATION: [specific location if YES, otherwise "N/A"]
DESCRIPTION: [brief 1-sentence description]

Be conservative - only say YES if you see a clear outpouching."""
    
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
            # Use context window analysis if enabled
            if use_context:
                context_result = analyze_with_context(slices, model, idx, modality="CTA")
                response = context_result['response']
            else:
                response = model.analyze_image(
                    slice_data['pixel_array'],
                    prompt=detection_prompt,
                    system_prompt="You are an expert neuroradiologist specializing in cerebrovascular imaging. Analyze carefully."
                )
            
            # Improved detection parsing - check structured output
            has_aneurysm = parse_aneurysm_response(response)
            
            if has_aneurysm:
                location = extract_location_from_response(response)
                confidence = extract_confidence_from_response(response)
                print(f"⚠️ POTENTIAL FINDING! [{confidence}] at {location}")
                results['findings'].append({
                    'slice_index': idx,
                    'slice_number': idx + 1,
                    'filename': slice_data['filename'],
                    'location': location,
                    'confidence': confidence,
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
        print("  python analyze_all_slices.py <path_to_dicom_folder> [sample_every] [--context]")
        print("  python analyze_all_slices.py <series_uid> [sample_every] [--context]")
        print("\nOptions:")
        print("  sample_every: Analyze every Nth slice (default: 10, use 1 for ALL)")
        print("  --context: Use 3-slice context window for better 3D analysis")
        print("\nExamples:")
        print("  python analyze_all_slices.py 1.2.826.0.1.3680043.8.498.10035643...")
        print("  python analyze_all_slices.py <uid> 5 --context")
        return
    
    input_path = sys.argv[1]
    
    # Parse arguments
    sample_every = 10
    use_context = False
    
    for arg in sys.argv[2:]:
        if arg == '--context':
            use_context = True
        elif arg.isdigit():
            sample_every = int(arg)
    
    # Determine if it's a folder path or series UID
    if os.path.isdir(input_path):
        print(f"\n📁 Loading from folder: {input_path}")
        slices = load_all_slices_from_folder(input_path)
    else:
        print(f"\n📁 Loading Series UID from dataset ZIP")
        slices = load_all_slices_from_zip(input_path)
    
    # Analyze
    results = analyze_all_slices(slices, sample_every=sample_every, use_context=use_context)
    
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
            location = finding.get('location', 'Unknown')
            confidence = finding.get('confidence', 'N/A')
            print(f"  📍 Slice {finding['slice_number']}: {finding['filename']}")
            print(f"     Location: {location} | Confidence: {confidence}")
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
        f.write(f"Context mode: {'Enabled' if use_context else 'Disabled'}\n")
        f.write(f"Findings: {len(results['findings'])}\n\n")
        
        if results['findings']:
            f.write("POTENTIAL ANEURYSM LOCATIONS:\n")
            f.write("-" * 40 + "\n")
            for finding in results['findings']:
                location = finding.get('location', 'Unknown')
                confidence = finding.get('confidence', 'N/A')
                f.write(f"\nSlice {finding['slice_number']}: {finding['filename']}\n")
                f.write(f"Location: {location} | Confidence: {confidence}\n")
                f.write(f"Response: {finding['response']}\n")
    
    print(f"\n✅ Report saved to: {output_file}")
    print("\n💡 TIPS for better accuracy:")
    print("   • Analyze ALL slices: python analyze_all_slices.py <uid> 1")
    print("   • Use 3D context:     python analyze_all_slices.py <uid> 5 --context")


if __name__ == "__main__":
    main()
