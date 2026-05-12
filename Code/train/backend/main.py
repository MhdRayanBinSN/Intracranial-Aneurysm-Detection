"""
FastAPI Backend for Intracranial Aneurysm Detection.

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
import re
import shutil
import sys
import uuid
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np
from PIL import Image



# Initialize FastAPI app
# app = FastAPI(...) is moved below to include lifespan

# CORS middleware moved to after app initialization

# Centralized path configuration — override via environment variables
DATA_ROOT = Path(os.environ.get("DATA_ROOT", r"C:\Users\Rayan\Desktop\Main Project"))

# Global variables
_medgemma_instance = None   # MedGemma LLM — loaded at startup
_biomedclip_filter = None   # BiomedCLIP pre-filter — loaded at startup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Location names for predictions (13 RSNA locations + 1 overall)
LOCATION_NAMES = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]


def _env_int(name: str, default: int) -> int:
    """Read a positive integer environment setting with a safe fallback."""
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


# MedGemma is the expensive stage. Keep this high enough for sensitivity, but
# bounded so a full CTA/MRA series does not take hours.
MAX_SLICES_FOR_MEDGEMMA = _env_int("MEDGEMMA_MAX_SLICES", 24)
BASELINE_SERIES_SAMPLES = _env_int("MEDGEMMA_BASELINE_SAMPLES", 6)


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _image_position_z(dcm):
    ipp = getattr(dcm, "ImagePositionPatient", None)
    try:
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
    except (TypeError, ValueError):
        pass
    return None


def _slice_sort_key(slice_data: Dict[str, Any]):
    """
    Sort a DICOM/NIfTI series by acquisition metadata, not upload/file-name order.
    RSNA DICOM filenames are SOPInstanceUIDs, so lexical order is anatomically random.
    """
    series_uid = slice_data.get("series_uid") or ""
    filename = slice_data.get("filename") or ""
    instance_number = slice_data.get("instance_number")
    slice_location = slice_data.get("slice_location")
    position_z = slice_data.get("image_position_z")

    if instance_number is not None:
        return (series_uid, 0, float(instance_number), filename)
    if slice_location is not None:
        return (series_uid, 1, float(slice_location), filename)
    if position_z is not None:
        return (series_uid, 2, float(position_z), filename)
    return (series_uid, 9, 0.0, filename)


def _sort_dicom_file_paths(paths: List[str]) -> List[str]:
    """Sort on-disk DICOM paths using metadata, with filename fallback."""
    try:
        import pydicom
    except Exception:
        return sorted(paths)

    keyed = []
    for path in paths:
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            keyed.append((
                _slice_sort_key({
                    "series_uid": str(getattr(dcm, "SeriesInstanceUID", "")),
                    "instance_number": _safe_float(getattr(dcm, "InstanceNumber", None)),
                    "slice_location": _safe_float(getattr(dcm, "SliceLocation", None)),
                    "image_position_z": _image_position_z(dcm),
                    "filename": Path(path).name,
                }),
                path,
            ))
        except Exception:
            keyed.append(((str(path), 9, 0.0, Path(path).name), path))
    return [path for _, path in sorted(keyed, key=lambda item: item[0])]


def _read_dicom_from_bytes(content: bytes):
    """Read uploaded bytes as DICOM, including valid DICOM files without .dcm."""
    import pydicom
    from io import BytesIO

    try:
        dcm = pydicom.dcmread(BytesIO(content))
    except Exception as first_error:
        try:
            dcm = pydicom.dcmread(BytesIO(content), force=True)
        except Exception:
            raise first_error

    if not hasattr(dcm, "PixelData"):
        raise ValueError("DICOM file has no PixelData")
    return dcm


def _dicom_to_slice(dcm, filename: str) -> Dict[str, Any]:
    arr = dcm.pixel_array.astype(np.float32)
    slope = float(getattr(dcm, 'RescaleSlope', 1))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
    arr = arr * slope + intercept

    return {
        "pixel_array": arr,
        "filename": filename,
        "original_filename": filename,
        "modality": None,  # filled by caller after modality detection
        "series_uid": str(getattr(dcm, "SeriesInstanceUID", "")),
        "sop_instance_uid": str(getattr(dcm, "SOPInstanceUID", "")),
        "instance_number": _safe_float(getattr(dcm, "InstanceNumber", None)),
        "slice_location": _safe_float(getattr(dcm, "SliceLocation", None)),
        "image_position_z": _image_position_z(dcm),
    }


def _simple_image_suspicion(image: Image.Image) -> float:
    """Fast fallback score when CLIP is unavailable; used only for slice ranking."""
    arr = np.array(image.convert("L")).astype(np.float32)
    if arr.size == 0:
        return 0.0
    p99 = float(np.percentile(arr, 99)) / 255.0
    bright_ratio = float(np.mean(arr > 220))
    return max(0.0, min(1.0, 0.7 * p99 + 0.3 * min(bright_ratio * 25.0, 1.0)))


def _estimate_candidate_bbox(image: Image.Image) -> List[int]:
    """
    Approximate a review box around the most focal bright candidate.
    This is not a medical segmentation mask; it avoids returning the whole image
    as the lesion box when the UI asks for a location hint.
    """
    arr = np.array(image.convert("L")).astype(np.float32)
    h, w = arr.shape
    if h == 0 or w == 0 or float(arr.max() - arr.min()) < 1e-6:
        return [0, 0, w, h]

    threshold = np.percentile(arr, 99.3)
    mask = arr >= threshold

    # Ignore the outer border where skull/table artifacts often dominate.
    y_margin = int(h * 0.06)
    x_margin = int(w * 0.06)
    mask[:y_margin, :] = False
    mask[h - y_margin:, :] = False
    mask[:, :x_margin] = False
    mask[:, w - x_margin:] = False

    try:
        from scipy import ndimage
        labeled, count = ndimage.label(mask)
        objects = ndimage.find_objects(labeled)
        best = None
        center = np.array([h / 2.0, w / 2.0])
        for label_idx, obj in enumerate(objects, start=1):
            if obj is None:
                continue
            ys, xs = obj
            area = int(np.sum(labeled[obj] == label_idx))
            if area < 3 or area > arr.size * 0.04:
                continue
            cy = (ys.start + ys.stop) / 2.0
            cx = (xs.start + xs.stop) / 2.0
            distance_penalty = np.linalg.norm(np.array([cy, cx]) - center) / max(h, w)
            intensity = float(np.mean(arr[obj][labeled[obj] == label_idx])) / 255.0
            score = intensity - 0.25 * distance_penalty
            if best is None or score > best[0]:
                best = (score, xs.start, ys.start, xs.stop, ys.stop)
        if best is not None:
            _, x1, y1, x2, y2 = best
        else:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any() or not cols.any():
                return [0, 0, w, h]
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
    except Exception:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return [0, 0, w, h]
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

    pad = max(8, int(min(h, w) * 0.025))
    return [
        int(max(0, x1 - pad)),
        int(max(0, y1 - pad)),
        int(min(w, x2 + pad)),
        int(min(h, y2 + pad)),
    ]


def _select_target_slices(all_slices: List[Dict[str, Any]], max_slices: int) -> List[Dict[str, Any]]:
    """
    Choose a bounded set for MedGemma while preserving recall:
    suspicious score peaks plus neighbors, with a few evenly spaced baseline slices.
    """
    n = len(all_slices)
    if n <= max_slices:
        return list(all_slices)

    selected = set()
    baseline_count = min(n, max(1, min(BASELINE_SERIES_SAMPLES, max_slices // 4)))
    for idx in np.linspace(0, n - 1, baseline_count, dtype=int):
        selected.add(int(idx))

    ranked = sorted(
        range(n),
        key=lambda i: float(all_slices[i].get("clip_score", 0.0)),
        reverse=True,
    )
    for center in ranked:
        for offset in (0, -1, 1):
            candidate = center + offset
            if 0 <= candidate < n:
                selected.add(candidate)
            if len(selected) >= max_slices:
                break
        if len(selected) >= max_slices:
            break

    return [all_slices[i] for i in sorted(selected)]


# Auto-cleanup: remove temp analysis images older than 1 hour
def _cleanup_old_temp_images():
    """Periodically clean up old analysis images from temp directory."""
    img_root = os.path.join(tempfile.gettempdir(), "medgemma_images")
    if not os.path.exists(img_root):
        return
    now = time.time()
    for entry in os.listdir(img_root):
        entry_path = os.path.join(img_root, entry)
        if os.path.isdir(entry_path):
            age = now - os.path.getmtime(entry_path)
            if age > 3600:  # older than 1 hour
                shutil.rmtree(entry_path, ignore_errors=True)


# Pydantic models for request/response
class PredictionResult(BaseModel):
    """Single prediction result."""
    location: str
    probability: float
    detected: bool


class AnalysisResponse(BaseModel):
    """Analysis response model."""
    id: str
    status: str
    predictions: List[PredictionResult]
    overall_risk: str
    confidence: float
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool


class SeriesInfo(BaseModel):
    """Information about a DICOM series."""
    series_uid: str
    patient_age: Optional[str]
    patient_sex: Optional[str]
    modality: str
    num_slices: int


# Store for analysis results
analysis_store: Dict[str, Any] = {}


# Lifespan context manager (Replaces deprecated on_event)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Handles model loading and cleanup.
    """
    global _medgemma_instance, _biomedclip_filter

    # --- STARTUP ---
    print("🚀 Starting Backend (MedGemma + BiomedCLIP)...", flush=True)

    # Shared medgemma path
    medgemma_path = str(Path(__file__).parent.parent / 'medgemma')
    if medgemma_path not in sys.path:
        sys.path.insert(0, medgemma_path)

    # 1. Load BiomedCLIP pre-filter first (small model, fast)
    try:
        print("🔬 Loading BiomedCLIP pre-filter (~500MB)...", flush=True)
        from biomedclip_filter import BiomedCLIPFilter
        _biomedclip_filter = BiomedCLIPFilter()
        print("✅ BiomedCLIP pre-filter ready!", flush=True)
    except Exception as e:
        print(f"⚠️ BiomedCLIP not available (will send ALL slices to MedGemma): {e}", flush=True)
        _biomedclip_filter = None

    # 2. Load MedGemma (large model, blocking — server starts AFTER ready)
    try:
        print("🔄 Loading MedGemma model (this takes 1-2 minutes)...", flush=True)
        from inference import MedGemmaInference
        _medgemma_instance = MedGemmaInference()
        print("✅ MedGemma model loaded and ready!", flush=True)
    except Exception as e:
        print(f"⚠️ Error loading MedGemma: {e}", flush=True)
        _medgemma_instance = None

    yield

    # --- SHUTDOWN ---
    print("🛑 Shutting down Backend...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Intracranial Aneurysm Detection API",
    description="API for detecting intracranial aneurysms from medical imaging",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5134"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modular routers ────────────────────────────────────────────
from dataset_router import router as dataset_router
app.include_router(dataset_router)
# ──────────────────────────────────────────────────────────────



@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=_medgemma_instance is not None,
        device=device,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Run temp file cleanup on health check
    _cleanup_old_temp_images()
    return HealthResponse(
        status="healthy",
        model_loaded=_medgemma_instance is not None,
        device=device,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/analyze")
async def analyze_dicom(
    files: List[UploadFile] = File(...),
    modality: str = "CTA",
):
    """
    Analyze DICOM files for aneurysm detection.
    Redirects to MedGemma pipeline (legacy ResNet3D model removed).
    """
    if _medgemma_instance is None:
        raise HTTPException(status_code=503, detail="MedGemma model not loaded yet")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Delegate to the MedGemma upload pipeline
    file_contents = [{"filename": f.filename, "content": f.file.read()} for f in files]
    return _run_medgemma_logic_full(file_contents)


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis result by ID."""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_store[analysis_id]


@app.get("/locations")
async def get_locations():
    """Get list of anatomical locations."""
    return {
        "locations": LOCATION_NAMES,
        "count": len(LOCATION_NAMES),
    }


@app.post("/demo/predict")
async def demo_predict():
    """
    Demo endpoint - returns mock predictions.
    Useful for frontend development without model.
    """
    import random
    
    predictions = []
    for name in LOCATION_NAMES:
        prob = random.random() * 0.3  # Low probabilities for most
        if "Middle Cerebral" in name:
            prob = random.random() * 0.8  # Higher for demo
        
        predictions.append(PredictionResult(
            location=name,
            probability=prob,
            detected=prob > 0.5,
        ))
    
    overall_prob = predictions[-1].probability
    risk = "High" if overall_prob > 0.7 else "Moderate" if overall_prob > 0.3 else "Low"
    
    return AnalysisResponse(
        id="demo-" + str(uuid.uuid4())[:4],
        status="completed",
        predictions=predictions,
        overall_risk=risk,
        confidence=overall_prob,
        processing_time=0.5,
    )


# ==================== MEDGEMMA ENDPOINTS ====================

# MedGemma response models
class MedGemmaFinding(BaseModel):
    """Single MedGemma finding."""
    slice_index: int
    slice_number: int
    response: str


class MedGemmaResponse(BaseModel):
    """MedGemma analysis response."""
    id: str
    status: str
    series_uid: str
    report: str
    slices_analyzed: int
    has_findings: bool
    findings: List[MedGemmaFinding]
    processing_time: float


# Store for MedGemma results
medgemma_store: Dict[str, Any] = {}


@app.get("/medgemma/patients")
async def list_medgemma_patients():
    """List patients from the dataset for MedGemma analysis."""
    import pandas as pd
    
    train_csv = DATA_ROOT / "train.csv"
    
    if not train_csv.exists():
        return {"success": False, "error": f"train.csv not found at {train_csv}", "patients": []}
    
    try:
        df = pd.read_csv(train_csv)
        
        # Get unique series - check which columns exist
        if 'SeriesInstanceUID' in df.columns:
            uid_col = 'SeriesInstanceUID'
        elif 'series_id' in df.columns:
            uid_col = 'series_id'
        else:
            return {"success": False, "error": f"No series column found. Columns: {list(df.columns)[:10]}", "patients": []}
        
        # Get modality if available
        if 'Modality' in df.columns:
            patients = df[[uid_col, 'Modality']].drop_duplicates().head(50)
            return {
                "success": True,
                "patients": [
                    {
                        "series_uid": str(row[uid_col]),
                        "modality": row['Modality'],
                        "display_name": f"{row['Modality']} - {str(row[uid_col])[:20]}..."
                    }
                    for _, row in patients.iterrows()
                ]
            }
        else:
            patients = df[[uid_col]].drop_duplicates().head(50)
            return {
                "success": True,
                "patients": [
                    {
                        "series_uid": str(row[uid_col]),
                        "modality": "CT",
                        "display_name": f"CT - {str(row[uid_col])[:25]}..."
                    }
                    for _, row in patients.iterrows()
                ]
            }
    except Exception as e:
        return {"success": False, "error": str(e), "patients": []}





@app.post("/medgemma/analyze-upload")
def analyze_uploaded_files(files: List[UploadFile] = File(...)):
    """
    Analyze uploaded DICOM/NIfTI slices with candidate selection and MedGemma.
    """
    # BUG FIX: Read all bytes FIRST before any processing touches the streams
    file_contents = [{"filename": f.filename, "content": f.file.read()} for f in files]
    return _run_medgemma_logic_full(file_contents)


def _run_medgemma_logic_full(file_contents: list) -> dict:
    """Core MedGemma processing logic — accepts pre-read file bytes."""
    import traceback
    from io import BytesIO
    from PIL import Image

    # Import modality helpers from inference module
    medgemma_path = str(Path(__file__).parent.parent / 'medgemma')
    if medgemma_path not in sys.path:
        sys.path.insert(0, medgemma_path)
    from inference import detect_modality_from_dicom, detect_modality_from_pixels, preprocess_for_modality

    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    if not file_contents:
        raise HTTPException(status_code=400, detail="No files uploaded")

    print(f"\n{'='*60}")
    print(f"📤 Received {len(file_contents)} files for analysis")
    print(f"{'='*60}")

    findings = []
    slices_analyzed = 0

    # Create temp directory for this analysis's images
    analysis_img_dir = os.path.join(tempfile.gettempdir(), "medgemma_images", analysis_id)
    os.makedirs(analysis_img_dir, exist_ok=True)

    # 1. Collect all valid slices from inputs (DICOM & NIfTI)
    all_slices = []

    for fc in file_contents:
        fname = fc["filename"].lower()
        content = fc["content"]
        file = type('MockFile', (), {'filename': fc["filename"]})()  # for error logging
        try:
            
            if fname.endswith(('.nii', '.nii.gz')):
                # Handle NIfTI Volume
                try:
                    import nibabel as nib
                    # NIfTI requires a file path usually
                    nii_suffix = ".nii.gz" if fname.endswith('.nii.gz') else ".nii"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=nii_suffix) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    
                    nii = nib.load(tmp_path)
                    data = nii.get_fdata()
                    os.unlink(tmp_path) # Cleanup immediate
                    
                    # Assume Z is last dimension (H, W, D)
                    if len(data.shape) == 3:
                        num_slices = data.shape[2]
                        # Limit to reasonable number if massive? For now take all.
                        for z in range(num_slices):
                            # Transpose if needed, but strict axial is tricky without orientation.
                            # Standardizing to float32
                            arr = data[:, :, z].astype(np.float32)
                            # Rotate if needed (NIfTI is often rotated relative to DICOM view)
                            # We'll assume standard orientation for now or fix in preprocessor
                            arr = np.rot90(arr) 
                            
                            # Detect modality from pixel stats (NIfTI has no DICOM tags)
                            sample_mid = data[:, :, num_slices // 2].astype(np.float32)
                            nii_mod = detect_modality_from_pixels(sample_mid)
                            all_slices.append({
                                "pixel_array": arr,
                                "filename": f"{fname} [Slice {z+1}]",
                                "original_filename": fname,
                                "modality": nii_mod,
                                "series_uid": fname,
                                "sop_instance_uid": f"{fname}:{z+1}",
                                "instance_number": z + 1,
                                "slice_location": z,
                                "image_position_z": z,
                            })
                except ImportError:
                    print("⚠️ nibabel not installed, skipping NIfTI")
                except Exception as e:
                    print(f"⚠️ NIfTI error {fname}: {e}")
                    
            else:
                # Handle DICOM regardless of extension. Some DICOM series use
                # numeric/no-extension filenames, and filtering by ".dcm" drops them.
                try:
                    dcm = _read_dicom_from_bytes(content)
                    modality = detect_modality_from_dicom(dcm)
                    slice_record = _dicom_to_slice(dcm, fname)
                    slice_record["modality"] = modality
                    all_slices.append(slice_record)
                except Exception as e:
                    print(f"⚠️ DICOM/medical image error {fname}: {e}")
                
        except Exception as e:
            print(f"Error reading {file.filename}: {e}")

    if not all_slices:
        raise HTTPException(
            status_code=400,
            detail="No valid DICOM or NIfTI slices found. Upload a complete medical image series.",
        )

    # Critical for correctness: upload order and SOPInstanceUID filename order are
    # not anatomical order. Sort before numbering, sampling, and reporting.
    all_slices.sort(key=_slice_sort_key)
    for i, slice_data in enumerate(all_slices):
        slice_data["original_idx"] = i
        slice_data["slice_number"] = i + 1

    # 2. Pre-filter with BiomedCLIP — select Top-K most suspicious slices
    # Without filtering: 200+ slices can take hours. Candidate selection keeps
    # top-scored slices, their neighbors, and a few baseline samples.
    global _biomedclip_filter, _medgemma_instance
    if _biomedclip_filter is not None and all_slices:
        _filter_start = time.time()
        print(f"  🔬 Scoring all {len(all_slices)} slices with BiomedCLIP...")

        modality_batches = defaultdict(list)
        for i, slice_data in enumerate(all_slices):
            modality = slice_data.get("modality", "CTA")
            region_img = preprocess_for_modality(slice_data["pixel_array"], modality)
            slice_data["region_img"] = region_img  # cache so we don't recompute
            modality_batches[modality].append((i, region_img))

        for modality, batch in modality_batches.items():
            images = [img for _, img in batch]
            try:
                clip_scores = _biomedclip_filter.batch_score(images, modality=modality)
            except Exception as e:
                print(f"  ⚠️ Batch CLIP scoring failed for {modality}; using per-slice scoring: {e}")
                clip_scores = [_biomedclip_filter.score(img, modality=modality) for img in images]

            for (slice_idx, img), clip_score in zip(batch, clip_scores):
                try:
                    intensity_score = _biomedclip_filter.intensity_heuristic(img)
                except Exception:
                    intensity_score = _simple_image_suspicion(img)
                all_slices[slice_idx]["clip_score"] = float(
                    max(0.0, min(1.0, 0.7 * float(clip_score) + 0.3 * float(intensity_score)))
                )

        target_slices = _select_target_slices(all_slices, MAX_SLICES_FOR_MEDGEMMA)
        kept_scores = [s.get("clip_score", 0.0) for s in target_slices]
        print(
            f"  🎯 Selected {len(target_slices)} of {len(all_slices)} slices "
            f"(score range: {min(kept_scores):.3f}-{max(kept_scores):.3f})"
        )
        print(f"  📍 Slice numbers sent to MedGemma: {[s['slice_number'] for s in target_slices]}")
        print(f"  ⏱️  BiomedCLIP scoring took {time.time() - _filter_start:.1f}s")
    else:
        print("  ⚠️ BiomedCLIP unavailable; using fast intensity ranking and series sampling")
        for slice_data in all_slices:
            modality = slice_data.get("modality", "CTA")
            region_img = preprocess_for_modality(slice_data["pixel_array"], modality)
            slice_data["region_img"] = region_img
            slice_data["clip_score"] = _simple_image_suspicion(region_img)
        target_slices = _select_target_slices(all_slices, MAX_SLICES_FOR_MEDGEMMA)
        print(f"  📍 Slice numbers sent to MedGemma: {[s['slice_number'] for s in target_slices]}")

    print(f"============================================================")
    print(f"🧠 MEDGEMMA ANALYSIS ON {len(target_slices)} SELECTED SLICES")
    print(f"============================================================")

    # 3. Analyze the target slices
    for loop_idx, slice_data in enumerate(target_slices):
        idx = slice_data["original_idx"]
        slice_number = slice_data.get("slice_number", idx + 1)
        clip_score = slice_data["clip_score"]
        slice_filename = slice_data["filename"]
        
        # Compatibility: Create 'file' object for except block
        class MockFile: pass
        file = MockFile()
        file.filename = slice_data.get("original_filename", slice_filename)
        # Ensure slice_file is available for the LLM step later
        slice_file = file
        
        try:
            pixel_array = slice_data["pixel_array"]
            slices_analyzed += 1
            h, w = pixel_array.shape

            # Resolve modality per-slice (each slice carries its own)
            modality = slice_data.get("modality", "CTA")

            # === MODALITY-SPECIFIC PREPROCESSING ===
            # Use cached region_img if available
            region_img = slice_data.get("region_img")
            if region_img is None:
                region_img = preprocess_for_modality(pixel_array, modality)
            
            windowed = np.array(region_img.convert("L"))  # For intensity fallback stats

            slice_info = f"Slice {slice_number} of {len(all_slices)}"
            if hasattr(slice_file, 'filename'):
                slice_info += f" ({slice_file.filename})"

            print(f"  🔬 [{modality}] Slice {slice_number} suspicious (score={clip_score:.3f}) → sending to MedGemma")

            # === MEDGEMMA DETAILED ANALYSIS (expensive ~30-120s) ===
            if _medgemma_instance is None:
                print(f"  ⚠️ Skipping Slice {slice_number}: MedGemma not loaded yet")
                continue

            medgemma = _medgemma_instance

            print(f"  🧠 Running MedGemma [{modality}] on {slice_info}...")
            _inference_start = time.time()
            llm_analysis = medgemma.analyze_for_kaggle(
                region_img,
                modality=modality,
                slice_info=slice_info,
                clip_score=clip_score,
                # CoT disabled for screening — takes 227s vs 40s on RTX 3060.
                # Fast structured prompt is sufficient for YES/NO screening.
                use_cot=False,
            )
            _inference_time = time.time() - _inference_start
            print(f"  ✅ MedGemma done in {_inference_time:.1f}s (BiomedCLIP score={clip_score:.3f})")

            # Parse MedGemma response to determine if finding is positive
            # Import the robust parser from analyze_all_slices
            location_extractor = None
            try:
                from analyze_all_slices import parse_aneurysm_response, extract_location_from_response
                location_extractor = extract_location_from_response
                is_positive = parse_aneurysm_response(llm_analysis)
            except ImportError:
                # Fallback to inline parsing if module not available
                llm_lower = llm_analysis.lower()
                is_positive = (
                    "aneurysm_detected: yes" in llm_lower or
                    "aneurysm detected: yes" in llm_lower or
                    ("confidence: high" in llm_lower and "yes" in llm_lower) or
                    "saccular aneurysm" in llm_lower or
                    ("outpouching" in llm_lower and "no" not in llm_lower[:50])
                )
                # Explicit NO overrides vague positive keywords
                if "aneurysm_detected: no" in llm_lower or "aneurysm: no" in llm_lower:
                    if "confidence: high" not in llm_lower:
                        is_positive = False


            if not is_positive:
                continue  # MedGemma found nothing suspicious — skip

            # === MAP SPATIAL POSITION TO ANATOMICAL LOCATION ===
            # MedGemma provides detailed location in its structured text response
            location_name = "Intracranial Vessel"  # fallback

            if location_extractor is not None:
                try:
                    parsed_loc = location_extractor(llm_analysis)
                    if parsed_loc and parsed_loc.lower() not in ("unknown location", "unknown", "n/a", "none"):
                        location_name = parsed_loc
                except Exception:
                    pass
            
            # Extract from MedGemma output format: "LOCATION: Right MCA"

            loc_match = re.search(r'LOCATION:\s*(.*?)(?:\n|$)', llm_analysis, re.IGNORECASE)
            if loc_match:
                parsed_loc = loc_match.group(1).strip()
                # Clean up if the model added brackets or extra text
                parsed_loc = parsed_loc.replace('[', '').replace(']', '')
                if parsed_loc and parsed_loc.lower() not in ('unknown', 'n/a', 'none', 'rsna location name'):
                    if location_name == "Intracranial Vessel":
                        location_name = parsed_loc

            # Save image for this finding
            img_path = os.path.join(analysis_img_dir, f"slice_{idx}.png")
            region_img.save(img_path, format="PNG")
            img_url = f"/medgemma/finding-image/{analysis_id}/{idx}"
            bbox = _estimate_candidate_bbox(region_img)

            response = f"""MEDGEMMA ANALYSIS — {location_name}

{llm_analysis}

Note: AI analysis — requires radiologist review."""

            print(f"  >> Slice {slice_number}: Finding at {location_name}")

            findings.append({
                "slice_index": idx,
                "slice_number": slice_number,
                "response": response,
                "image": img_url,
                "regions": 1,
                "bbox": bbox,
                "location": location_name,
                "intensity": float(np.mean(windowed))
            })

        except Exception as e:
            print(f"Error processing file {slice_data.get('filename', '?')}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    # Group findings by anatomical location
    findings_by_location = defaultdict(list)
    for f in findings:
        loc = f.get("location", "Unknown")
        findings_by_location[loc].append({
            "slice_index": f["slice_index"],
            "slice_number": f["slice_number"],
            "response": f["response"],
            "image": f["image"],  # URL, not base64
            "bbox": f["bbox"],
            "intensity": f.get("intensity", 0),
            "regions": f.get("regions", 0),
        })
    
    # Sort slices within each location by slice number
    for loc in findings_by_location:
        findings_by_location[loc].sort(key=lambda x: x["slice_number"])
    
    # Count unique locations
    num_locations = len(findings_by_location)
    total_detection_slices = len(findings)
    
    if findings:
        total_regions = sum(f.get("regions", 0) for f in findings)
        report = f"""CT SCAN ANALYSIS REPORT

Analyzed: {slices_analyzed} slice(s)
Findings: {total_detection_slices} slice(s) with detections across {num_locations} location(s)
Total ROIs: {total_regions} region(s) of interest

Locations with findings:
{chr(10).join(f'  - {loc}: {len(slices)} slice(s)' for loc, slices in findings_by_location.items())}

Detection Method: MedGemma vision-language analysis with candidate slice preselection

This is automated detection. All findings require verification by a qualified radiologist."""
    else:
        report = f"""CT SCAN ANALYSIS REPORT

Analyzed: {slices_analyzed} slice(s)
No MedGemma-positive aneurysm findings detected in the selected candidate slices.

Detection Method: MedGemma vision-language analysis with candidate slice preselection

Note: Absence of findings does not rule out pathology."""
    
    try:
        result = {
            "id": analysis_id,
            "status": "completed",
            "report": report,
            "slices_analyzed": slices_analyzed,
            "has_findings": len(findings) > 0,
            "findings": findings,
            "findings_by_location": dict(findings_by_location),
            "num_locations": num_locations,
            "processing_time": processing_time,
        }
        print(f"\n{'='*60}")
        print(f"✅ Analysis complete! {len(findings)} findings in {processing_time:.1f}s")
        print(f"   Returning response with {len(findings)} findings across {num_locations} locations")
        print(f"{'='*60}\n")
        return result
    except Exception as e:
        print(f"\n❌ RESPONSE BUILD ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


# Serve full-quality slice images on demand (no base64 = no memory issues)
@app.get("/medgemma/finding-image/{analysis_id}/{slice_index}")
async def get_finding_image(analysis_id: str, slice_index: int):
    """Serve a processed slice image as a full-quality PNG file."""
    img_path = os.path.join(tempfile.gettempdir(), "medgemma_images", analysis_id, f"slice_{slice_index}.png")
    
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(img_path, media_type="image/png")



@app.post("/medgemma/analyze")
async def analyze_medgemma(
    series_uid: str,
    sample_every: int = 10,
):
    """
    Analyze an extracted dataset series with the same real pipeline used for uploads.
    sample_every is kept for API compatibility; candidate selection happens in
    _run_medgemma_logic_full so the full series can still be ranked correctly.
    """
    import glob
    
    series_folder = DATA_ROOT / "series" / series_uid
    
    if not series_folder.exists():
        raise HTTPException(status_code=404, detail=f"Series folder not found: {series_folder}")
    
    try:
        dcm_files = sorted(glob.glob(str(series_folder / "*.dcm")))
        if not dcm_files:
            dcm_files = sorted([str(f) for f in series_folder.iterdir() if f.is_file()])
        
        if not dcm_files:
            raise HTTPException(status_code=404, detail="No DICOM files found in series folder")

        dcm_files = _sort_dicom_file_paths(dcm_files)
        file_contents = [
            {"filename": Path(path).name, "content": Path(path).read_bytes()}
            for path in dcm_files
        ]

        print(f"📸 Running real MedGemma analysis for {len(file_contents)} slices from {series_uid[:20]}...")
        result = _run_medgemma_logic_full(file_contents)
        result["series_uid"] = series_uid
        medgemma_store[result["id"]] = result
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MedGemma analysis failed: {str(e)}")


@app.get("/medgemma/results/{analysis_id}")
async def get_medgemma_results(analysis_id: str):
    """Get MedGemma analysis results by ID."""
    if analysis_id not in medgemma_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return medgemma_store[analysis_id]


@app.get("/medgemma/slice/{series_uid}/{slice_index}")
async def get_slice_image(series_uid: str, slice_index: int, heatmap: bool = False):
    """
    Get a slice image, optionally with heatmap overlay.
    Returns base64-encoded PNG image.
    """
    import pydicom
    import base64
    from io import BytesIO
    from PIL import Image
    import glob
    
    series_folder = DATA_ROOT / "series" / series_uid
    
    if not series_folder.exists():
        raise HTTPException(status_code=404, detail="Series not found")
    
    # Get DICOM files
    dcm_files = sorted(glob.glob(str(series_folder / "*.dcm")))
    if not dcm_files:
        dcm_files = sorted([str(f) for f in series_folder.iterdir() if f.is_file()])
    dcm_files = _sort_dicom_file_paths(dcm_files)
    
    if slice_index >= len(dcm_files):
        raise HTTPException(status_code=404, detail="Slice index out of range")
    
    try:
        # Read DICOM
        dcm = pydicom.dcmread(dcm_files[slice_index])
        pixel_array = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, 'RescaleSlope', 1))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0))
        pixel_array = pixel_array * slope + intercept
        
        # Apply CTA windowing
        window_center, window_width = 300, 600
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        windowed = np.clip(pixel_array, lower, upper)
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        
        # Convert to RGB
        rgb = np.stack([windowed, windowed, windowed], axis=-1)
        
        # Add simple heatmap overlay if requested
        if heatmap:
            # Create a simple center-focused heatmap (simulated attention)
            h, w = windowed.shape
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            # Create radial gradient
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            heatmap_mask = 1 - (dist / max_dist)
            heatmap_mask = (heatmap_mask * 0.5).clip(0, 1)
            
            # Apply red overlay
            rgb[:, :, 0] = np.minimum(255, rgb[:, :, 0] + (heatmap_mask * 100).astype(np.uint8))
        
        # Create PIL image
        img = Image.fromarray(rgb)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "slice_index": slice_index,
            "total_slices": len(dcm_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading slice: {str(e)}")




# ==================== COMPARE ENDPOINT ====================

@app.post("/compare")
def compare_models(files: List[UploadFile] = File(...)):
    """
    Run both models on the same uploaded files and return side-by-side results.
    - MedGemma: candidate slice selection + vision-language analysis
    - Legacy ML: ResNet3D-18 (currently unavailable - no trained checkpoint)
    """
    import time
    start = time.time()

    # BUG FIX: Read all file bytes ONCE into memory before any processing.
    # UploadFile streams can only be consumed once — reading them here
    # prevents a 'stream already consumed' error if we called two pipelines.
    file_contents = [{"filename": f.filename, "content": f.file.read()} for f in files]

    # --- RUN MEDGEMMA PIPELINE ---
    medgemma_result = None
    try:
        medgemma_result = _run_medgemma_logic_full(file_contents)
    except Exception as e:
        medgemma_result = {
            "available": False,
            "error": str(e),
            "has_findings": False,
            "slices_analyzed": 0,
            "findings": [],
            "findings_by_location": {},
            "num_locations": 0,
        }

    # --- LEGACY ML MODEL ---
    # Confirmed unavailable: train/checkpoints/ is empty (no best.pt found).
    # The model also requires a full 32-slice volume [128x128x32] — a single
    # DICOM file is insufficient. Show an informative unavailable card instead.
    legacy_result = {
        "available": False,
        "error": "Legacy model not trained yet",
        "info": (
            "Requires trained ResNet3D-18 weights at ml/checkpoints/best.pt "
            "and a full 32-slice DICOM series (128x128 px). "
            "Train the model first using ml/train.py."
        ),
        "overall_risk": None,
        "confidence": None,
        "predictions": [],
    }

    return {
        "medgemma": medgemma_result,
        "legacy": legacy_result,
        "comparison_time": round(time.time() - start, 2),
    }


if __name__ == "__main__":
    port = int(os.environ.get("BACKEND_PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disabled — reload causes deadlocks with heavy model loading
    )

