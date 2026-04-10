"""
MedGemma Inference Pipeline
Multi-modality support: CTA, MRA, MRI T2, MRI T1 Post

Based on official HuggingFace docs: https://huggingface.co/google/medgemma-4b-it
"""

import gc
import os

# CRITICAL: Disable safetensors memory mapping to avoid "paging file too small" error
os.environ["SAFETENSORS_FAST_GPU"] = "0"

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Union
from transformers.utils import logging
logging.set_verbosity_warning()
logging.disable_progress_bar()

from config import (
    MODEL_ID, HF_TOKEN, CACHE_DIR,
    MAX_NEW_TOKENS, MAX_NEW_TOKENS_COT,
    SYSTEM_PROMPT, ANALYSIS_PROMPT,
    SYSTEM_PROMPT_CTA, SYSTEM_PROMPT_MRA,
    SYSTEM_PROMPT_MRI_T2, SYSTEM_PROMPT_MRI_T1_POST,
    CT_WINDOW_CENTER, CT_WINDOW_WIDTH,
    get_analysis_prompt,
)


# ==============================================================
# MODALITY DETECTION
# ==============================================================

def detect_modality_from_dicom(dcm) -> str:
    """
    Detect imaging modality from DICOM metadata.
    Returns one of: 'CTA', 'MRA', 'MRI_T2', 'MRI_T1_POST', 'CT', 'MRI'
    """
    modality_tag = str(getattr(dcm, 'Modality', '')).upper()
    series_desc  = str(getattr(dcm, 'SeriesDescription', '')).upper()
    study_desc   = str(getattr(dcm, 'StudyDescription', '')).upper()
    protocol     = str(getattr(dcm, 'ProtocolName', '')).upper()
    sequence     = str(getattr(dcm, 'ScanningSequence', '')).upper()
    contrast     = str(getattr(dcm, 'ContrastBolusAgent', '')).upper()

    combined = f"{series_desc} {study_desc} {protocol} {sequence} {contrast}"

    # --- CT branch ---
    if modality_tag == 'CT':
        if 'ANGIO' in combined or 'CTA' in combined or 'ANGIOGRAPHY' in combined:
            return 'CTA'
        if contrast and contrast not in ('NONE', 'NO', ''):
            return 'CTA'
        return 'CT'

    # --- MR branch ---
    if modality_tag == 'MR':
        # MRA (MR Angiography)
        if any(k in combined for k in ('ANGIO', 'MRA', 'TOF', 'TIME OF FLIGHT', 'ANGIOGRAPHY')):
            return 'MRA'
        # T1 post-contrast
        if any(k in combined for k in ('POST', 'CONTRAST', 'GADOLINIUM', 'GAD', 'T1 +', 'T1+')):
            return 'MRI_T1_POST'
        if contrast and contrast not in ('NONE', 'NO', ''):
            return 'MRI_T1_POST'
        # T2-weighted
        if 'T2' in combined:
            return 'MRI_T2'
        # Default MR → T2 (most common brain MR)
        return 'MRI_T2'

    # Fallback — treat as CTA (most common in dataset)
    return 'CTA'


def detect_modality_from_pixels(pixel_array: np.ndarray) -> str:
    """
    Heuristic modality detection from pixel statistics when DICOM metadata unavailable.
    CT/CTA has HU values (can be negative, range ~-1000 to +3000).
    MRI intensity is arbitrary but always >= 0.
    """
    arr = pixel_array.astype(np.float32)
    has_negatives = np.any(arr < -50)
    p99 = np.percentile(arr, 99)

    if has_negatives or p99 > 500:
        # Strong indicator of CT Hounsfield units
        # With contrast (bright vessels up to ~400 HU) → CTA
        p_vessel = np.percentile(arr[arr > 100], 95) if np.any(arr > 100) else 0
        if p_vessel > 200:
            return 'CTA'
        return 'CT'
    else:
        # MRI — no negatives, bounded range
        return 'MRI_T2'  # Safe default


# ==============================================================
# MODALITY-SPECIFIC PREPROCESSING
# ==============================================================

def preprocess_for_modality(pixel_array: np.ndarray, modality: str) -> Image.Image:
    """
    Convert raw pixel array to 3-channel RGB PIL Image using modality-appropriate settings.

    - CTA/CT: Multi-window (Brain | Vessel | Bone) → preserves contrast and vessel visibility
    - MRA: Percentile normalization → preserves bright vessel signal
    - MRI T2: Percentile normalization → keeps CSF bright, vessels dark (flow void)
    - MRI T1 Post: Percentile normalization → enhancement pattern preserved
    """
    arr = pixel_array.astype(np.float32)
    mod = modality.upper()

    if mod in ('CTA', 'CT'):
        # Multi-window RGB encoding (like Kaggle 1st place)
        # Ch0 = Brain window,  Ch1 = Vessel window,  Ch2 = Bone window
        windows = [(40, 80), (300, 600), (700, 3000)]
        channels = []
        for center, width in windows:
            lo = center - width / 2
            hi = center + width / 2
            ch = np.clip(arr, lo, hi)
            ch = ((ch - lo) / (hi - lo) * 255).astype(np.uint8)
            channels.append(ch)
        rgb = np.stack(channels, axis=-1)

    elif mod == 'MRA':
        # Percentile normalization: vessel signal (bright) preserved
        lo = np.percentile(arr, 0.5)
        hi = np.percentile(arr, 99.5)
        if hi - lo < 1e-6:
            hi = lo + 1
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
        ch = (norm * 255).astype(np.uint8)
        rgb = np.stack([ch, ch, ch], axis=-1)

    elif mod in ('MRI_T2', 'MRI T2'):
        # Percentile normalization for T2: CSF bright, vessels dark
        lo = np.percentile(arr, 0.5)
        hi = np.percentile(arr, 99.5)
        if hi - lo < 1e-6:
            hi = lo + 1
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
        ch = (norm * 255).astype(np.uint8)
        # Store as grey — T2 is inherently greyscale; colour adds confusion
        rgb = np.stack([ch, ch, ch], axis=-1)

    elif mod in ('MRI_T1_POST', 'MRI T1 POST', 'T1', 'T1_POST'):
        # Percentile normalization for T1 post-contrast
        lo = np.percentile(arr, 0.5)
        hi = np.percentile(arr, 99.5)
        if hi - lo < 1e-6:
            hi = lo + 1
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
        ch = (norm * 255).astype(np.uint8)
        rgb = np.stack([ch, ch, ch], axis=-1)

    else:
        # Fallback — use CTA vessel window
        lo, hi = 300 - 300, 300 + 300  # lo=0, hi=600
        ch = np.clip(arr, lo, hi)
        denom = hi - lo
        if denom < 1e-6:
            denom = 1.0
        ch = ((ch - lo) / denom * 255).astype(np.uint8)
        rgb = np.stack([ch, ch, ch], axis=-1)

    return Image.fromarray(rgb)


def get_system_prompt(modality: str) -> str:
    """Return the appropriate system prompt for the given modality."""
    mod = modality.upper()
    if mod == 'MRA':
        return SYSTEM_PROMPT_MRA
    elif mod in ('MRI_T2', 'MRI T2'):
        return SYSTEM_PROMPT_MRI_T2
    elif mod in ('MRI_T1_POST', 'MRI T1 POST', 'T1', 'T1_POST'):
        return SYSTEM_PROMPT_MRI_T1_POST
    else:  # CTA, CT, unknown
        return SYSTEM_PROMPT_CTA


# ==============================================================
# MAIN INFERENCE CLASS
# ==============================================================

class MedGemmaInference:
    """
    MedGemma inference class for multi-modality medical image analysis.
    Supports CTA, MRA, MRI T2, MRI T1 Post-contrast.

    Usage:
        model = MedGemmaInference()
        result = model.analyze_slice(pixel_array, modality="MRA")
        print(result)
    """

    def __init__(self, model_id: str = MODEL_ID, offline_mode: bool = True):
        """Initialize MedGemma model with 4-bit quantization on GPU."""
        print("=" * 60, flush=True)
        print("🏥 MEDGEMMA - Multi-Modality Medical AI", flush=True)
        print("=" * 60, flush=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Device: {self.device}", flush=True)

        print(f"📥 Loading model: {model_id}", flush=True)
        if offline_mode:
            print("   (Offline mode - using cached model)", flush=True)

        self.use_pipeline = False
        self.pipe = None

        if offline_mode:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

        # Free memory before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_vram = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"🧹 Cleared cache. Free VRAM: {free_vram:.2f} GB", flush=True)

        # 4-bit quantization: 8GB → ~2.5GB
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )
        print("🚀 Using 4-bit quantization (8GB → ~2.5GB)", flush=True)

        try:
            max_memory = {0: "4GB", "cpu": "10GB"}
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                token=HF_TOKEN,
            )
            self.processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)

            if torch.cuda.is_available():
                free_after = torch.cuda.mem_get_info()[0] / 1024**3
                print(f"✅ Model loaded on GPU! Free VRAM: {free_after:.2f} GB", flush=True)
            else:
                print("✅ Model loaded successfully!", flush=True)

        except Exception as e:
            print(f"⚠️  4-bit GPU loading failed: {e}", flush=True)
            print("📦 Falling back to CPU loading...", flush=True)
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="cpu",
                    token=HF_TOKEN,
                )
                self.processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
                print("✅ Model loaded on CPU (slower inference)", flush=True)
            except Exception as e2:
                print(f"❌ Error loading model: {e2}", flush=True)
                raise e2

    def _run_inference(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str,
        max_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        """Core inference — sends image + text to MedGemma and returns response."""
        messages = [
            {"role": "system",  "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",    "content": [
                {"type": "text",  "text": prompt},
                {"type": "image", "image": image},
            ]},
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                max_length=None,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True)

    def analyze_slice(
        self,
        pixel_array: np.ndarray,
        modality: str = "CTA",
        slice_info: str = "",
    ) -> str:
        """
        Main entry point: preprocesses a raw pixel array using modality-specific
        windowing and sends it to MedGemma with the appropriate clinical prompt.

        Args:
            pixel_array: Raw 2D numpy array (HU values for CT, arbitrary for MRI)
            modality:    One of 'CTA', 'CT', 'MRA', 'MRI_T2', 'MRI_T1_POST'
            slice_info:  Human-readable slice identifier for logging

        Returns:
            MedGemma's analysis text
        """
        image = preprocess_for_modality(pixel_array, modality)
        system_prompt = get_system_prompt(modality)
        prompt = get_analysis_prompt(modality, slice_info)
        response = self._run_inference(image, prompt, system_prompt)
        return f"### MedGemma Analysis [{modality}]\n\n{response}"

    def analyze_image(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        modality: str = "CTA",
        prompt: str = None,
        system_prompt: str = None,
    ) -> str:
        """
        Analyze a pre-processed image (PIL or path).
        For raw pixel arrays use analyze_slice() instead.
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = preprocess_for_modality(image, modality)

        sys_p = system_prompt or get_system_prompt(modality)
        usr_p = prompt or get_analysis_prompt(modality)
        return self._run_inference(image, usr_p, sys_p)

    def analyze_for_kaggle(
        self,
        image: Union[Image.Image, np.ndarray],
        modality: str = "CTA",
        slice_info: str = "Unknown Slice",
        clip_score: float = None,
        use_cot: bool = True,
    ) -> str:
        """
        Main analysis entry point used by the backend.
        Accepts a PIL Image or ndarray and runs modality-aware analysis
        with chain-of-thought reasoning.

        Args:
            image:      PIL Image or raw numpy array
            modality:   Imaging modality (CTA, MRA, MRI_T2, MRI_T1_POST)
            slice_info: Slice identifier for logging
            clip_score: BiomedCLIP suspicion score (for confidence ensemble)
            use_cot:    If True, uses chain-of-thought prompt for better reasoning
        """
        if isinstance(image, np.ndarray):
            image = preprocess_for_modality(image, modality)

        system_prompt = get_system_prompt(modality)

        if use_cot:
            prompt = self._get_cot_prompt(modality, slice_info)
        else:
            prompt = get_analysis_prompt(modality, slice_info)

        response = self._run_inference(
            image, prompt, system_prompt,
            max_tokens=MAX_NEW_TOKENS_COT if use_cot else MAX_NEW_TOKENS,
        )

        # Build confidence label using ensemble (BiomedCLIP score + MedGemma verdict)
        confidence_label = self._ensemble_confidence(response, clip_score)

        return (
            f"### MedGemma RSNA Analysis [{modality}] — {confidence_label}\n\n"
            f"{response}"
        )

    def analyze_context_window(
        self,
        pixel_arrays: list,
        center_idx: int,
        modality: str = "CTA",
        slice_info: str = "",
        clip_score: float = None,
    ) -> str:
        """
        Accuracy Improvement #1 — Multi-Slice Context Window.

        Sends 3 consecutive slices (prev, current, next) as a single grid image
        to MedGemma. This gives 3D context: an aneurysm visible across multiple
        slices is real; a single-slice bright spot is likely noise.

        Args:
            pixel_arrays: List of ALL raw pixel arrays in the scan
            center_idx:   Index of the slice to analyze (N-1, N, N+1 used)
            modality:     Imaging modality
            slice_info:   Slice identifier

        Returns:
            MedGemma's analysis with multi-slice context
        """
        # Collect N-1, N, N+1 (clamp to valid range)
        n = len(pixel_arrays)
        indices = [
            max(0, center_idx - 1),
            center_idx,
            min(n - 1, center_idx + 1),
        ]

        # Preprocess each slice
        imgs = [preprocess_for_modality(pixel_arrays[i], modality) for i in indices]

        # Build side-by-side grid
        grid = self.build_context_grid(imgs, labels=[
            f"Slice {indices[0]+1}",
            f"Slice {indices[1]+1} ← FOCUS",
            f"Slice {indices[2]+1}",
        ])

        system_prompt = get_system_prompt(modality)
        prompt = self._get_cot_prompt_multiview(modality, slice_info, indices)
        response = self._run_inference(grid, prompt, system_prompt)
        confidence_label = self._ensemble_confidence(response, clip_score)

        return (
            f"### MedGemma 3-Slice Context Analysis [{modality}] — {confidence_label}\n\n"
            f"{response}"
        )

    # ----------------------------------------------------------
    # HELPER: Build 3-slice grid image (Improvement #1)
    # ----------------------------------------------------------
    @staticmethod
    def build_context_grid(
        images: list,
        labels: list = None,
        target_height: int = 336,
    ) -> Image.Image:
        """
        Stitch a list of PIL Images side-by-side into a single grid image.
        Optionally draws slice labels at the top of each panel.

        Args:
            images:        List of PIL Images (2 or 3)
            labels:        Optional list of label strings per image
            target_height: Resize each panel to this height

        Returns:
            Single wide PIL Image
        """
        from PIL import ImageDraw, ImageFont

        panels = []
        for img in images:
            # Resize to standard height while keeping aspect ratio
            ratio  = target_height / img.height
            new_w  = int(img.width * ratio)
            panel  = img.resize((new_w, target_height), Image.LANCZOS)
            panel  = panel.convert("RGB")
            panels.append(panel)

        # Determine total width
        total_w = sum(p.width for p in panels)
        grid = Image.new("RGB", (total_w, target_height), color=(20, 20, 20))

        x_offset = 0
        for i, panel in enumerate(panels):
            grid.paste(panel, (x_offset, 0))
            # Add label
            if labels and i < len(labels):
                draw = ImageDraw.Draw(grid)
                draw.text((x_offset + 4, 4), labels[i], fill=(255, 255, 0))
            x_offset += panel.width

        return grid

    # ----------------------------------------------------------
    # HELPER: Chain-of-Thought prompts (Improvement #2)
    # ----------------------------------------------------------
    def _get_cot_prompt(self, modality: str, slice_info: str = "") -> str:
        """
        Generate a chain-of-thought prompt that forces MedGemma to reason
        step-by-step through vessel anatomy before giving a final answer.
        This reduces hallucinations and improves specificity.
        """
        mod = modality.upper()
        slice_str = f" ({slice_info})" if slice_info else ""

        # Modality-specific vessel appearance for CoT
        vessel_appearance = {
            "CTA":        "bright white contrast-filled",
            "CT":         "slightly bright",
            "MRA":        "bright (TOF signal)",
            "MRI_T2":     "dark flow voids",
            "MRI_T1_POST":"strongly enhancing",
        }.get(mod, "visible")

        aneurysm_appearance = {
            "CTA":        "bright rounded outpouching at a bifurcation",
            "CT":         "hyperdense rounded lesion",
            "MRA":        "bright rounded protrusion from vessel",
            "MRI_T2":     "rounded signal void or mixed signal at bifurcation",
            "MRI_T1_POST":"uniformly enhancing or rim-enhancing rounded mass",
        }.get(mod, "rounded outpouching")

        return f"""You are analyzing a {modality} brain scan{slice_str} for intracranial aneurysms.
Vessels appear {vessel_appearance}. An aneurysm looks like a {aneurysm_appearance}.

Reason step-by-step before concluding:

STEP 1 — VESSELS VISIBLE:
List every vessel segment you can identify (ICA, MCA, ACA, PComm, Basilar, etc.)

STEP 2 — BIFURCATION INSPECTION:
For each vessel, describe what you see at its bifurcation or branching point.
Is there any focal outpouching, bulge, or shape irregularity?

STEP 3 — CANDIDATE ASSESSMENT:
If any candidate was found in Step 2:
  - Is it truly saccular (≥2mm, attached to artery wall)?
  - Or is it an infundibulum (<3mm funnel), vascular loop, or artifact?

STEP 4 — FINAL ANSWER:
ANEURYSM_DETECTED: YES / NO
If YES:
  LOCATION: [RSNA location — e.g. Left MCA, Basilar Tip, AComm]
  SIZE_ESTIMATE: [small <5mm / medium 5-15mm / large >15mm]
  MORPHOLOGY: [saccular / fusiform / blister]
  CONFIDENCE: HIGH / MEDIUM / LOW
  DESCRIPTION: [2-3 sentences]
If NO:
  REASON: [normal vessels / artifact / image quality / etc.]"""

    def _get_cot_prompt_multiview(
        self, modality: str, slice_info: str, indices: list
    ) -> str:
        """Chain-of-thought prompt adapted for 3-slice grid view."""
        slice_labels = f"Slices {indices[0]+1}, {indices[1]+1}, {indices[2]+1}"
        base = self._get_cot_prompt(modality, slice_info)
        header = (
            f"You are looking at a 3-SLICE CONTEXT GRID showing {slice_labels} "
            f"side-by-side. The MIDDLE panel is the primary slice to analyze; "
            f"use the LEFT and RIGHT panels for 3D structural context.\n\n"
            f"A true aneurysm will appear as a consistent 3D structure across "
            f"multiple consecutive slices — not just a single-slice bright spot.\n\n"
        )
        return header + base

    # ----------------------------------------------------------
    # HELPER: Confidence Ensemble (Improvement #3)
    # ----------------------------------------------------------
    @staticmethod
    def _ensemble_confidence(medgemma_response: str, clip_score: float = None) -> str:
        """
        Accuracy Improvement #3 — Confidence-Weighted Ensemble.

        Combines BiomedCLIP pre-filter score (if available) with MedGemma's
        verdict to produce a final confidence label:

          CLIP HIGH + MedGemma YES  → ✅ HIGH CONFIDENCE
          CLIP LOW  + MedGemma YES  → ⚠️ MEDIUM CONFIDENCE (review)
          CLIP HIGH + MedGemma NO   → 🔍 FLAG FOR REVIEW  (possible miss)
          CLIP LOW  + MedGemma NO   → ✅ CONFIRMED NEGATIVE
        """
        resp_lower = medgemma_response.lower()
        gemma_positive = any(kw in resp_lower for kw in [
            "aneurysm_detected: yes",
            "aneurysm detected: yes",
            "yes\n", ": yes",
            "saccular aneurysm",
            "outpouching",
            "bulge",
        ])

        if clip_score is None:
            # No CLIP score — just use MedGemma
            return "✅ POSITIVE" if gemma_positive else "🔵 NEGATIVE"

        clip_suspicious = clip_score >= 0.30

        if clip_suspicious and gemma_positive:
            return f"✅ HIGH CONFIDENCE (CLIP={clip_score:.2f})"
        elif not clip_suspicious and gemma_positive:
            return f"⚠️ MEDIUM CONFIDENCE — review (CLIP={clip_score:.2f}, low for positive)"
        elif clip_suspicious and not gemma_positive:
            return f"🔍 FLAG FOR REVIEW (CLIP={clip_score:.2f} suspicious but MedGemma negative)"
        else:
            return f"🔵 CONFIRMED NEGATIVE (CLIP={clip_score:.2f})"


    def analyze_dicom(self, dicom_path: Union[str, Path]) -> str:
        """Analyze a DICOM file — auto-detects modality from metadata."""
        import pydicom

        try:
            dcm = pydicom.dcmread(str(dicom_path))
            modality = detect_modality_from_dicom(dcm)

            pixel_array = dcm.pixel_array.astype(np.float32)
            slope = float(getattr(dcm, 'RescaleSlope', 1))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
            pixel_array = pixel_array * slope + intercept

            print(f"📋 DICOM modality detected: {modality}")
            return self.analyze_slice(pixel_array, modality=modality)

        except Exception as e:
            return f"Error analyzing DICOM: {str(e)}"


# Simple test
if __name__ == "__main__":
    print("\n🧪 Testing MedGemma Multi-Modality Pipeline...")
    if HF_TOKEN is None:
        print("\n❌ Error: HF_TOKEN not set!")
    else:
        try:
            model = MedGemmaInference()
            print("\n✅ MedGemma is ready!")
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
