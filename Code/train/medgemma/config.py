"""
MedGemma Configuration
Multi-modality support: CTA, MRA, MRI T2, MRI T1 Post
"""
import os
from pathlib import Path

# Model Settings
MODEL_ID = "google/medgemma-4b-it"  # 4B Multimodal — approved & cached locally

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
DATASET_ZIP = os.environ.get(
    "DATASET_ZIP",
    r"C:\Users\Rayan\Desktop\Main Project\rsna-intracranial-aneurysm-detection.zip"
)

# Inference Settings
MAX_NEW_TOKENS     = 200   # Fast structured prompt — enough for YES/NO + location
MAX_NEW_TOKENS_COT = 400   # Chain-of-thought — needs more room for step-by-step reasoning
TEMPERATURE = 0.7
TOP_P = 0.9

# ============================================================
# MODALITY-SPECIFIC PREPROCESSING WINDOWS
# ============================================================
# CTA — CT Angiography: vessels filled with contrast (bright)
# Uses multi-window: Brain | Vessel | Bone
CTA_WINDOWS = [
    (40,  80),    # Brain window  — gray/white matter
    (300, 600),   # Vessel window — opacified arteries bright
    (700, 3000),  # Bone window   — skull base / calcification
]

# MRA — MR Angiography: Time-of-Flight or contrast, no HU units
# Intensity is arbitrary; percentile normalization used instead
MRA_WINDOWS = None  # Signal percentile normalization (0.5–99.5%)

# MRI T2 — T2-weighted: CSF bright, vessels dark (flow void)
# Good for SAH (subarachnoid hemorrhage) detection
MRI_T2_WINDOWS = None  # Percentile normalization (0.5–99.5%)

# MRI T1 Post — T1 with gadolinium contrast: enhancing lesions bright
# Aneurysm walls and thrombus may enhance
MRI_T1_POST_WINDOWS = None  # Percentile normalization (0.5–99.5%)

# Legacy default CT window (kept for backward compat)
CT_WINDOW_CENTER = 300
CT_WINDOW_WIDTH  = 600

# ============================================================
# MODALITY-SPECIFIC SYSTEM PROMPTS
# ============================================================

SYSTEM_PROMPT_CTA = """You are an expert neuroradiologist analyzing a Brain CT Angiography (CTA) scan for intracranial aneurysms.

CTA characteristics: Arteries appear bright white due to iodine contrast. Look for rounded outpouchings from vessel walls.

DETECTION TARGET: Saccular (berry) aneurysms — round/lobulated outpouchings at vessel bifurcations ≥2mm.
EXCLUDE: Infundibula (<3mm funnel shapes), fusiform aneurysms, vascular loops.

13 RSNA ANATOMICAL LOCATIONS TO CHECK:
1. Left Infraclinoid ICA    2. Right Infraclinoid ICA
3. Left Supraclinoid ICA   4. Right Supraclinoid ICA
5. Left MCA                6. Right MCA
7. Anterior Communicating  8. Left ACA   9. Right ACA
10. Left PComm             11. Right PComm
12. Basilar Tip            13. Other Posterior Circulation

Be SPECIFIC and CONSERVATIVE. High specificity is critical."""

SYSTEM_PROMPT_MRA = """You are an expert neuroradiologist analyzing a Brain MR Angiography (MRA) scan for intracranial aneurysms.

MRA characteristics: Flowing blood appears bright (Time-of-Flight or contrast-enhanced). Vessel walls are dark. Aneurysms appear as bright focal outpouchings from vessel lumen.

DETECTION TARGET: Saccular aneurysms at vessel bifurcations. On MRA these appear as rounded bright structures projecting from parent vessels.
IMPORTANT: MRA has lower spatial resolution than CTA. Smaller aneurysms (<3mm) may be missed.

13 RSNA ANATOMICAL LOCATIONS TO CHECK:
1. Left Infraclinoid ICA    2. Right Infraclinoid ICA
3. Left Supraclinoid ICA   4. Right Supraclinoid ICA
5. Left MCA                6. Right MCA
7. Anterior Communicating  8. Left ACA   9. Right ACA
10. Left PComm             11. Right PComm
12. Basilar Tip            13. Other Posterior Circulation

Report with confidence level: HIGH (definite outpouching), MEDIUM (possible), LOW (uncertain)."""

SYSTEM_PROMPT_MRI_T2 = """You are an expert neuroradiologist analyzing a Brain MRI T2-weighted scan for intracranial aneurysms.

T2 MRI characteristics: CSF appears bright (white), brain parenchyma is grey, vessels show as signal voids (dark). Aneurysms appear as rounded signal voids (flow void phenomenon). Thrombosed aneurysms may show mixed signal.
SAH signs: Look for T2 hypointensity (darkness) in CSF spaces suggesting subarachnoid hemorrhage.

DETECTION TARGET: Flow voids in unusual locations (bifurcations), T2-hypointense rounded structures adjacent to vessels, SAH signs in basal cisterns.

13 RSNA ANATOMICAL LOCATIONS TO CHECK:
1. Left Infraclinoid ICA    2. Right Infraclinoid ICA
3. Left Supraclinoid ICA   4. Right Supraclinoid ICA
5. Left MCA                6. Right MCA
7. Anterior Communicating  8. Left ACA   9. Right ACA
10. Left PComm             11. Right PComm
12. Basilar Tip            13. Other Posterior Circulation"""

SYSTEM_PROMPT_MRI_T1_POST = """You are an expert neuroradiologist analyzing a post-contrast Brain MRI T1-weighted scan for intracranial aneurysms.

T1 Post-contrast characteristics: Gadolinium-enhancing structures appear bright. Patent aneurysm lumen enhances uniformly. Thrombosed portions do NOT enhance (appear dark). Aneurysm wall may show rim enhancement.

DETECTION TARGET: Rounded enhancing structures at vessel bifurcations. Non-enhancing (thrombosed) aneurysms appear as rounded structures with rim enhancement. Peri-aneurysmal enhancement may indicate inflammatory aneurysm.

13 RSNA ANATOMICAL LOCATIONS TO CHECK:
1. Left Infraclinoid ICA    2. Right Infraclinoid ICA
3. Left Supraclinoid ICA   4. Right Supraclinoid ICA
5. Left MCA                6. Right MCA
7. Anterior Communicating  8. Left ACA   9. Right ACA
10. Left PComm             11. Right PComm
12. Basilar Tip            13. Other Posterior Circulation"""

# General fallback prompt (when modality unknown)
SYSTEM_PROMPT = SYSTEM_PROMPT_CTA

# ============================================================
# MODALITY-SPECIFIC ANALYSIS PROMPTS
# ============================================================

def get_analysis_prompt(modality: str, slice_info: str = "") -> str:
    """Generate an analysis prompt tailored to the imaging modality."""
    mod = modality.upper()
    slice_str = f" ({slice_info})" if slice_info else ""

    if mod in ("CTA", "CT"):
        return f"""Analyze this CTA brain scan{slice_str} for intracranial aneurysms.

The arteries appear WHITE due to contrast. Look at all visible bifurcations for rounded outpouchings.
Do not output any reasoning or steps. Output EXACTLY this format and nothing else.

Answer format:
MODALITY: CTA
ANEURYSM_DETECTED: YES / NO
If YES:
  LOCATION: [RSNA location name]
  SIDE: Left / Right / Midline
  SIZE_ESTIMATE: [small <5mm / medium 5-15mm / large >15mm]
  MORPHOLOGY: [saccular / fusiform / blister]
  CONFIDENCE: HIGH / MEDIUM / LOW
  DESCRIPTION: [brief clinical description]
If NO:
  REASON: [normal vessels / poor image quality / etc.]"""

    elif mod == "MRA":
        return f"""Analyze this MRA brain scan{slice_str} for intracranial aneurysms.

Flowing blood appears BRIGHT. Look for bright rounded outpouchings at bifurcations.
Note: MRA has lower resolution than CTA. Focus on structures clearly different from normal anatomy.
Do not output any reasoning or steps. Output EXACTLY this format and nothing else.

Answer format:
MODALITY: MRA
ANEURYSM_DETECTED: YES / NO
If YES:
  LOCATION: [RSNA location name]
  SIDE: Left / Right / Midline
  SIZE_ESTIMATE: [small <5mm / medium 5-15mm / large >15mm]
  MORPHOLOGY: [saccular / fusiform]
  CONFIDENCE: HIGH / MEDIUM / LOW
  DESCRIPTION: [note MRA-specific features]
If NO:
  REASON: [normal / limited by resolution / etc.]"""

    elif mod in ("MRI_T2", "MRI T2", "T2"):
        return f"""Analyze this T2-weighted MRI brain scan{slice_str} for intracranial aneurysms.

CSF is BRIGHT WHITE. Vessels are DARK (flow void). Aneurysms appear as dark rounded structures at bifurcations.
Also check for SAH: T2-dark signal in normally bright CSF spaces (subarachnoid cisterns).
Do not output any reasoning or steps. Output EXACTLY this format and nothing else.

Answer format:
MODALITY: MRI T2
ANEURYSM_DETECTED: YES / NO
SAH_SIGNS: YES / NO
If YES:
  LOCATION: [RSNA location name]
  SIDE: Left / Right / Midline
  SIZE_ESTIMATE: [small / medium / large]
  T2_SIGNAL: [signal void / mixed signal - thrombosed]
  CONFIDENCE: HIGH / MEDIUM / LOW
  DESCRIPTION: [brief description]
If NO:
  REASON: [normal flow voids / etc.]"""

    elif mod in ("MRI_T1_POST", "MRI T1 POST", "T1", "T1_POST"):
        return f"""Analyze this post-contrast T1-weighted MRI brain scan{slice_str} for intracranial aneurysms.

Enhancing structures appear BRIGHT WHITE. Patent aneurysms enhance uniformly. Thrombosed portions appear dark.
Look for: bright rounded masses at bifurcations, rim-enhancing structures, peri-aneurysmal enhancement.
Do not output any reasoning or steps. Output EXACTLY this format and nothing else.

Answer format:
MODALITY: MRI T1 Post-contrast
ANEURYSM_DETECTED: YES / NO
If YES:
  LOCATION: [RSNA location name]
  SIDE: Left / Right / Midline
  SIZE_ESTIMATE: [small / medium / large]
  ENHANCEMENT_PATTERN: [uniform / rim only / no enhancement - thrombosed]
  CONFIDENCE: HIGH / MEDIUM / LOW
  DESCRIPTION: [brief description]
If NO:
  REASON: [normal enhancement pattern / etc.]"""

    else:
        return get_analysis_prompt("CTA", slice_info)  # Fallback


# Legacy constant (used by old code paths)
ANALYSIS_PROMPT = get_analysis_prompt("CTA")
