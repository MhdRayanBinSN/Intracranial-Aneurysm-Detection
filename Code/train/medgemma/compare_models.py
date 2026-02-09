"""
Compare ResNet vs MedGemma
Side-by-side comparison of both approaches.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))

from config import DATASET_ZIP, OUTPUT_DIR


def compare_models():
    """
    Compare ResNet and MedGemma on the same CT scan.
    """
    print("=" * 70)
    print("🔬 MODEL COMPARISON: ResNet vs MedGemma")
    print("=" * 70)
    
    # Load sample CT
    from utils.image_loader import load_dicom_from_zip, ct_to_pil
    
    print("\n📂 Loading sample CT scan...")
    try:
        ct_slice, patient_id = load_dicom_from_zip(DATASET_ZIP)
        print(f"✅ Patient: {patient_id[:30]}...")
    except Exception as e:
        print(f"❌ Error loading CT: {e}")
        return
    
    # Convert to image
    ct_image = ct_to_pil(ct_slice)
    
    # ==========================================
    # RESNET PREDICTION
    # ==========================================
    print("\n" + "=" * 40)
    print("🧠 MODEL 1: 3D ResNet (Your Current Model)")
    print("=" * 40)
    
    resnet_result = None
    try:
        from models.model import AneurysmDetector
        import torch
        
        # Note: ResNet needs 3D volume, not 2D slice
        # This is a simplified demo
        print("⚠️  Note: ResNet needs full 3D volume, using mock prediction")
        resnet_result = {
            'aneurysm_present': 0.75,
            'locations': {
                'Left_ICA': 0.65,
                'Right_MCA': 0.32,
                'Basilar': 0.15
            }
        }
        print("📊 ResNet Output (Mock):")
        print(f"   Aneurysm Present: {resnet_result['aneurysm_present']*100:.1f}%")
        for loc, prob in resnet_result['locations'].items():
            print(f"   {loc}: {prob*100:.1f}%")
            
    except Exception as e:
        print(f"⚠️  ResNet not available: {e}")
        resnet_result = {"error": str(e)}
    
    # ==========================================
    # MEDGEMMA PREDICTION
    # ==========================================
    print("\n" + "=" * 40)
    print("🏥 MODEL 2: MedGemma (Google's Medical AI)")
    print("=" * 40)
    
    medgemma_result = None
    try:
        from inference import MedGemmaInference
        
        model = MedGemmaInference()
        medgemma_result = model.analyze_image(ct_slice)
        
        print("📋 MedGemma Output:")
        print("-" * 40)
        print(medgemma_result)
        
    except Exception as e:
        print(f"⚠️  MedGemma not available: {e}")
        print("   (You may need to set HF_TOKEN)")
        medgemma_result = f"Error: {e}"
    
    # ==========================================
    # COMPARISON VISUALIZATION
    # ==========================================
    print("\n" + "=" * 40)
    print("📊 Creating Comparison Visualization...")
    print("=" * 40)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CT Image
    axes[0].imshow(ct_image)
    axes[0].set_title("Input: CT Angiography\n(Brain Scan)", fontweight='bold')
    axes[0].axis('off')
    
    # ResNet Output
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].text(5, 8, "3D ResNet", fontsize=14, ha='center', fontweight='bold')
    axes[1].text(5, 6, "Output: Probabilities", fontsize=10, ha='center')
    if isinstance(resnet_result, dict) and 'aneurysm_present' in resnet_result:
        axes[1].text(5, 4, f"Aneurysm: {resnet_result['aneurysm_present']*100:.1f}%", 
                    fontsize=12, ha='center')
        y = 2
        for loc, prob in resnet_result.get('locations', {}).items():
            axes[1].text(5, y, f"{loc}: {prob*100:.0f}%", fontsize=9, ha='center')
            y -= 0.8
    axes[1].set_title("Model 1: ResNet\n(Your Current)", fontweight='bold', color='blue')
    axes[1].axis('off')
    
    # MedGemma Output
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, 10)
    axes[2].text(5, 8, "MedGemma", fontsize=14, ha='center', fontweight='bold')
    axes[2].text(5, 6, "Output: Natural Language", fontsize=10, ha='center')
    if medgemma_result and not medgemma_result.startswith("Error"):
        # Truncate for display
        short_result = medgemma_result[:150] + "..." if len(str(medgemma_result)) > 150 else medgemma_result
        axes[2].text(5, 3, short_result, fontsize=8, ha='center', wrap=True,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        axes[2].text(5, 3, "(Requires HF_TOKEN)", fontsize=10, ha='center', color='red')
    axes[2].set_title("Model 2: MedGemma\n(Google's Medical AI)", fontweight='bold', color='green')
    axes[2].axis('off')
    
    plt.suptitle(f"Aneurysm Detection: Model Comparison\nPatient: {patient_id[:30]}...", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = OUTPUT_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"\n✅ Comparison saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("📌 SUMMARY")
    print("=" * 70)
    print("""
    | Feature          | 3D ResNet           | MedGemma              |
    |------------------|---------------------|----------------------|
    | Output           | Numbers (0-100%)    | Text Report          |
    | Training         | Needs YOUR data     | Pre-trained          |
    | Explainability   | Low (black box)     | High (explains why)  |
    | Best For         | Precise detection   | Clinical reports     |
    """)


if __name__ == "__main__":
    compare_models()
