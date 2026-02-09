"""
MedGemma Configuration
"""
import os
from pathlib import Path

# Model Settings
MODEL_ID = "google/medgemma-4b-it"  # 4B Multimodal (best for images)
# Alternative: "google/medgemma-27b-text-it" for text-only

# Hugging Face Token (required for gated models)
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset (same as main project)
DATASET_ZIP = r"C:\Users\Rayan\Desktop\Main Project\rsna-intracranial-aneurysm-detection.zip"

# Inference Settings
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

# CT Processing - CTA-OPTIMIZED for vessels
CT_WINDOW_CENTER = 300  # CTA vessel window (blood vessels appear bright)
CT_WINDOW_WIDTH = 600   # Wide enough to see vessel walls

# Alternative windows for different views
CTA_VESSEL_WINDOW = (300, 600)   # Best for seeing vessels
CTA_BRAIN_WINDOW = (40, 80)     # For brain parenchyma
CTA_BONE_WINDOW = (700, 3000)   # For skull base

# Prompts - IMPROVED for better accuracy
SYSTEM_PROMPT = """You are an expert neuroradiologist AI analyzing brain CT angiography (CTA) scans for intracranial aneurysms.

IMPORTANT CRITERIA FOR ANEURYSM DETECTION:
- Aneurysms appear as round/oval outpouchings from vessel walls
- They are typically 2-25mm in size
- Common locations (Circle of Willis):
  * Anterior communicating artery (AComm) - most common
  * Internal carotid artery (ICA) - at bifurcation
  * Middle cerebral artery (MCA) - M1 segment
  * Posterior communicating artery (PComm)
  * Basilar tip

WHAT IS NOT AN ANEURYSM:
- Normal vessel curves or loops
- Infundibulum (< 3mm funnel-shaped origin of vessels)
- Image artifacts or noise

Be CONSERVATIVE - only report findings with high confidence."""

# Quick detection prompt for slice scanning
QUICK_DETECT_PROMPT = """Examine this CTA slice carefully.

Is there a DEFINITE intracranial aneurysm visible?
- Look for a round/oval outpouching from a vessel
- Must be attached to a visible artery
- Must be ≥ 2mm

Answer format:
FINDING: YES or NO
LOCATION: (if YES) which artery
CONFIDENCE: HIGH, MEDIUM, or LOW
DESCRIPTION: Brief description"""

# Detailed analysis prompt
ANALYSIS_PROMPT = """Analyze this brain CTA slice for intracranial aneurysm.

Step-by-step analysis:
1. Identify visible vessels (ICA, MCA, ACA, basilar)
2. Check each vessel for outpouchings
3. If found, describe: location, size estimate, shape

Report your findings in this format:
VESSELS_VISIBLE: [list vessels you can see]
ANEURYSM_DETECTED: YES or NO
If YES:
  - LOCATION: [specific artery and side]
  - SIZE: [estimated mm or small/medium/large]
  - CONFIDENCE: [HIGH/MEDIUM/LOW]
  - DESCRIPTION: [brief description]
If NO:
  - REASON: [why no aneurysm visible]"""

