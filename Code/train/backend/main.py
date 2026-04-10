"""
FastAPI Backend for Intracranial Aneurysm Detection.

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
import shutil
import sys
import uuid



# Initialize FastAPI app
# app = FastAPI(...) is moved below to include lifespan

# CORS middleware moved to after app initialization

# Global variables
_medgemma_instance = None   # MedGemma LLM — loaded at startup
_biomedclip_filter = None   # BiomedCLIP pre-filter — loaded at startup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Location names for predictions
LOCATION_NAMES = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]


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
        model_loaded=model is not None,
        device=device,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=device,
        gpu_available=torch.cuda.is_available(),
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_dicom(
    files: List[UploadFile] = File(...),
    modality: str = "CTA",
):
    """
    Analyze DICOM files for aneurysm detection.
    
    Upload multiple DICOM files (one series) for analysis.
    """
    import time
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())[:8]
    
    # Create temporary directory for DICOM files
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Save uploaded files
        for f in files:
            file_path = temp_dir / f.filename
            with open(file_path, 'wb') as buffer:
                shutil.copyfileobj(f.file, buffer)
        
        # Preprocess
        volume, metadata = preprocessor.preprocess(str(temp_dir), modality)
        
        # Convert to tensor
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        volume_tensor = volume_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(volume_tensor)
            probabilities = outputs['probabilities'].cpu().numpy()[0]
        
        # Create predictions
        predictions = []
        threshold = 0.5
        
        for i, (name, prob) in enumerate(zip(LOCATION_NAMES, probabilities)):
            predictions.append(PredictionResult(
                location=name,
                probability=float(prob),
                detected=bool(prob > threshold),
            ))
        
        # Determine overall risk
        overall_prob = probabilities[-1]  # Last one is "Aneurysm Present"
        if overall_prob > 0.7:
            risk = "High"
        elif overall_prob > 0.3:
            risk = "Moderate"
        else:
            risk = "Low"
        
        processing_time = time.time() - start_time
        
        # Store result
        result = AnalysisResponse(
            id=analysis_id,
            status="completed",
            predictions=predictions,
            overall_risk=risk,
            confidence=float(overall_prob),
            processing_time=processing_time,
        )
        
        analysis_store[analysis_id] = result
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


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

# MedGemma model (lazy loaded)
medgemma_model = None
medgemma_model_error = None


def get_medgemma():
    """Lazy load MedGemma model with error handling."""
    global medgemma_model, medgemma_model_error
    
    # If we already tried and failed, return the error
    if medgemma_model_error:
        raise HTTPException(status_code=500, detail=f"MedGemma model failed to load: {medgemma_model_error}")
    
    if medgemma_model is None:
        try:
            print("🔄 Loading MedGemma model...")
            sys.path.insert(0, str(Path(__file__).parent.parent / 'medgemma'))
            from inference import MedGemmaInference
            medgemma_model = MedGemmaInference()
            print("✅ MedGemma model loaded!")
        except Exception as e:
            medgemma_model_error = str(e)
            print(f"❌ MedGemma loading failed: {e}")
            raise HTTPException(status_code=500, detail=f"MedGemma model failed to load: {e}")
    
    return medgemma_model


@app.get("/medgemma/patients")
async def list_medgemma_patients():
    """List patients from the dataset for MedGemma analysis."""
    import pandas as pd
    
    # Dataset path - using extracted folder
    data_root = Path("C:/Users/Rayan/Desktop/Main Project")
    train_csv = data_root / "train.csv"
    
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


def _run_medgemma_logic(file_contents: list) -> dict:
    """
    Core MedGemma processing logic — accepts pre-read file bytes.
    file_contents: list of {"filename": str, "content": bytes}
    Returns the same dict shape as /medgemma/analyze-upload.
    """
    import time
    import base64
    import traceback
    from io import BytesIO
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from scipy.ndimage import label, find_objects

    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    if not file_contents:
        raise HTTPException(status_code=400, detail="No files uploaded")

    print(f"\n{'='*60}")
    print(f"📤 Processing {len(file_contents)} files for MedGemma analysis")
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
        file = type('MockFile', (), {'filename': fc["filename"]})()  # for logging
        try:
            pass  # content already read
        except Exception:
            pass

        try:
            if fname.endswith(('.nii', '.nii.gz')):
                try:
                    import nibabel as nib
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii" if fname.endswith('.nii') else ".nii.gz") as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    nii = nib.load(tmp_path)
                    data = nii.get_fdata()
                    os.unlink(tmp_path)
                    if len(data.shape) == 3:
                        num_slices = data.shape[2]
                        # Detect modality from first slice pixels (NIfTI has no DICOM tags)
                        sample = data[:, :, num_slices // 2].astype(np.float32)
                        nii_modality = detect_modality_from_pixels(sample)
                        for z in range(num_slices):
                            arr = data[:, :, z].astype(np.float32)
                            arr = np.rot90(arr)
                            all_slices.append({
                                "pixel_array": arr,
                                "filename": f"{fname} [Slice {z+1}]",
                                "original_filename": fname,
                                "modality": nii_modality,
                            })
                except ImportError:
                    print("⚠️ nibabel not installed, skipping NIfTI")
                except Exception as e:
                    print(f"⚠️ NIfTI error {fname}: {e}")
            elif fname.endswith('.dcm'):
                try:
                    import pydicom
                    from io import BytesIO as _BIO
                    dcm = pydicom.dcmread(_BIO(content))
                    arr = dcm.pixel_array.astype(np.float32)
                    slope = float(getattr(dcm, 'RescaleSlope', 1))
                    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                    arr = arr * slope + intercept
                    all_slices.append({"pixel_array": arr, "filename": fname, "original_filename": fname})
                except Exception as e:
                    print(f"⚠️ DICOM error {fname}: {e}")
        except Exception as e:
            print(f"Error reading {fname}: {e}")

    # ---- from here the logic is identical to the original endpoint ----


@app.post("/medgemma/analyze-upload")
def analyze_uploaded_files(files: List[UploadFile] = File(...)):
    """
    Analyze uploaded CT slices with bounding box and heatmap visualization.
    Uses intensity-based region detection to find high-density areas.
    """
    # BUG FIX: Read all bytes FIRST before any processing touches the streams
    file_contents = [{"filename": f.filename, "content": f.file.read()} for f in files]
    return _run_medgemma_logic_full(file_contents)


def _run_medgemma_logic_full(file_contents: list) -> dict:
    import time
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
                    with tempfile.NamedTemporaryFile(delete=False, suffix=fname) as tmp:
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
                            
                            all_slices.append({
                                "pixel_array": arr,
                                "filename": f"{fname} [Slice {z+1}]",
                                "original_filename": fname
                            })
                except ImportError:
                    print("⚠️ nibabel not installed, skipping NIfTI")
                except Exception as e:
                    print(f"⚠️ NIfTI error {fname}: {e}")
                    
            elif fname.endswith('.dcm'):
                # Handle DICOM — detect modality from metadata
                try:
                    import pydicom
                    dcm = pydicom.dcmread(BytesIO(content))
                    # Detect modality from DICOM tags
                    modality = detect_modality_from_dicom(dcm)
                    # Rescale pixel values (HU for CT, arbitrary for MRI)
                    arr = dcm.pixel_array.astype(np.float32)
                    slope = float(getattr(dcm, 'RescaleSlope', 1))
                    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                    arr = arr * slope + intercept

                    all_slices.append({
                        "pixel_array": arr,
                        "filename": fname,
                        "original_filename": fname,
                        "modality": modality,
                    })
                except Exception as e:
                    print(f"⚠️ DICOM error {fname}: {e}")
            else:
                # Skip non-medical inputs (JPG/PNG) as requested
                # print(f"Skipping non-medical file: {fname}")
                pass
                
        except Exception as e:
            print(f"Error reading {file.filename}: {e}")

    if not all_slices:
        # Fallback for empty or all-invalid
        pass

    # 2. Pre-filter with BiomedCLIP to select Top 15 most suspicious slices
    # This prevents timeouts (228 slices = 2 hours) and guarantees we don't
    # accidentally filter out the whole scan due to uncalibrated thresholds.
    global _biomedclip_filter, _medgemma_instance
    if _biomedclip_filter is not None and all_slices:
        import time as _t
        _filter_start = _t.time()
        print(f"  🔬 Scoring all {len(all_slices)} slices with BiomedCLIP...")
        
        for i, slice_data in enumerate(all_slices):
            slice_data["original_idx"] = i
            modality = slice_data.get("modality", "CTA")
            region_img = preprocess_for_modality(slice_data["pixel_array"], modality)
            
            # Score every slice
            _, score = _biomedclip_filter.is_suspicious(region_img, modality=modality, threshold=0.0)
            slice_data["clip_score"] = score
            slice_data["region_img"] = region_img  # cache so we don't recompute
            
        # Don't filter out any slices because zero-shot BiomedCLIP might miss the aneurysm.
        # Just compute the score for ensemble logging.
        # Since MedGemma now runs at <1s per slice, we can safely process all slices.
        target_slices = all_slices
        
        print(f"  🎯 Kept all {len(target_slices)} slices. Scoring took {_t.time() - _filter_start:.1f}s")
        
        # Re-sort back to original anatomical order for sequential processing
        target_slices.sort(key=lambda x: x["original_idx"])
    else:
        for i, slice_data in enumerate(all_slices):
            slice_data["original_idx"] = i
            slice_data["clip_score"] = 1.0
        target_slices = all_slices

    print(f"============================================================")
    print(f"🧠 MEDGEMMA ANALYSIS ON ALL {len(target_slices)} SLICES")
    print(f"============================================================")

    # 3. Analyze the target slices
    for loop_idx, slice_data in enumerate(target_slices):
        idx = slice_data["original_idx"]
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

            # === MODALITY-SPECIFIC PREPROCESSING ===
            # Use cached region_img if available
            region_img = slice_data.get("region_img")
            if region_img is None:
                region_img = preprocess_for_modality(pixel_array, modality)
            
            windowed = np.array(region_img.convert("L"))  # For intensity fallback stats

            slice_info = f"Slice {idx+1}"
            if hasattr(slice_file, 'filename'):
                slice_info += f" ({slice_file.filename})"

            print(f"  🔬 [{modality}] Slice {idx+1} suspicious (score={clip_score:.3f}) → sending to MedGemma")

            # === MEDGEMMA DETAILED ANALYSIS (expensive ~30s) ===

            # === MEDGEMMA DETAILED ANALYSIS (expensive ~30-120s) ===
            if _medgemma_instance is None:
                print(f"  ⚠️ Skipping Slice {idx+1}: MedGemma not loaded yet")
                continue

            medgemma = _medgemma_instance

            print(f"  🧠 Running MedGemma [{modality}] on {slice_info}...")
            import time as _t
            _inference_start = _t.time()
            llm_analysis = medgemma.analyze_for_kaggle(
                region_img,
                modality=modality,
                slice_info=slice_info,
                clip_score=clip_score,
                # CoT only for highly suspicious slices — ~3-4x slower but more accurate
                use_cot=(clip_score is not None and clip_score > 0.60),
            )
            _inference_time = _t.time() - _inference_start
            print(f"  ✅ MedGemma done in {_inference_time:.1f}s (BiomedCLIP score={clip_score:.3f})")

            # Parse MedGemma response to determine if finding is positive
            # Import the robust parser from analyze_all_slices
            try:
                from analyze_all_slices import parse_aneurysm_response, extract_location_from_response, extract_confidence_from_response
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
            
            # Extract from MedGemma output format: "LOCATION: Right MCA"
            import re
            loc_match = re.search(r'LOCATION:\s*(.*?)(?:\n|$)', llm_analysis, re.IGNORECASE)
            if loc_match:
                parsed_loc = loc_match.group(1).strip()
                # Clean up if the model added brackets or extra text
                parsed_loc = parsed_loc.replace('[', '').replace(']', '')
                if parsed_loc and parsed_loc.lower() not in ('unknown', 'n/a', 'none', 'rsna location name'):
                    location_name = parsed_loc

            # Save image for this finding
            img_path = os.path.join(analysis_img_dir, f"slice_{idx}.png")
            region_img.save(img_path, format="PNG")
            img_url = f"/medgemma/finding-image/{analysis_id}/{idx}"

            response = f"""MEDGEMMA ANALYSIS — {location_name}

{llm_analysis}

Note: AI analysis — requires radiologist review."""

            print(f"  >> Slice {idx+1}: Finding at {location_name}")

            findings.append({
                "slice_index": idx,
                "slice_number": idx + 1,
                "response": response,
                "image": img_url,
                "regions": 1,
                "bbox": (0, 0, w, h),
                "location": location_name,
                "intensity": float(np.mean(windowed))
            })

        except Exception as e:
            print(f"Error processing file {slice_data.get('filename', '?')}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    # Group findings by anatomical location
    from collections import defaultdict
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

Detection Method: Intensity-based region segmentation

This is automated detection. All findings require verification by a qualified radiologist."""
    else:
        report = f"""CT SCAN ANALYSIS REPORT

Analyzed: {slices_analyzed} slice(s)
No significant high-intensity regions detected.

Detection Method: Intensity-based region segmentation

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
    Analyze a patient - DEMO MODE (no AI model required).
    
    Uses simple image analysis to generate demo findings for presentation.
    """
    import time
    import pydicom
    import glob
    import random
    
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]
    
    # Dataset path - using extracted folder
    data_root = Path("C:/Users/Rayan/Desktop/Main Project")
    series_folder = data_root / "series" / series_uid
    
    if not series_folder.exists():
        raise HTTPException(status_code=404, detail=f"Series folder not found: {series_folder}")
    
    try:
        # Load slices from folder
        dcm_files = sorted(glob.glob(str(series_folder / "*.dcm")))
        
        # If no .dcm extension, try all files
        if not dcm_files:
            dcm_files = sorted([str(f) for f in series_folder.iterdir() if f.is_file()])
        
        if not dcm_files:
            raise HTTPException(status_code=404, detail="No DICOM files found in series folder")
        
        total_slices = len(dcm_files)
        print(f"📸 Analyzing {total_slices} slices from series {series_uid[:20]}...")
        
        # Sample slices for analysis
        sample_indices = list(range(0, len(dcm_files), sample_every))
        findings = []
        
        # DEMO MODE: Analyze slices using simple image processing
        for idx in sample_indices:
            dcm_file = dcm_files[idx]
            
            try:
                dcm = pydicom.dcmread(dcm_file)
                pixel_array = dcm.pixel_array.astype(np.float32)
                slope = float(getattr(dcm, 'RescaleSlope', 1))
                intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                pixel_array = pixel_array * slope + intercept
                
                # DEMO: Simple intensity-based "detection" for demonstration
                # Checks for high-intensity regions that could indicate vessels/contrast
                
                # Apply CTA windowing
                window_center, window_width = 300, 600
                lower = window_center - window_width / 2
                upper = window_center + window_width / 2
                windowed = np.clip(pixel_array, lower, upper)
                
                # Calculate statistics
                mean_intensity = np.mean(windowed)
                max_intensity = np.max(windowed)
                high_intensity_ratio = np.sum(windowed > (window_center + window_width * 0.3)) / windowed.size
                
                # Demo detection logic (for demonstration purposes)
                # This is NOT real detection - just for demo/presentation
                is_interesting_slice = (
                    high_intensity_ratio > 0.02 and  # Has bright regions
                    mean_intensity > lower + window_width * 0.3  # Not too dark
                )
                
                # Add some randomness for demo variety (30% of interesting slices get flagged)
                if is_interesting_slice and random.random() > 0.7:
                    # Generate demo finding
                    locations = ["Internal Carotid Artery (ICA)", "Middle Cerebral Artery (MCA)", 
                                "Anterior Communicating Artery (AComm)", "Basilar Artery"]
                    sides = ["Left", "Right", ""]
                    confidences = ["HIGH", "MEDIUM"]
                    
                    location = random.choice(locations)
                    side = random.choice(sides)
                    confidence = random.choice(confidences)
                    
                    response = f"""FINDING: YES
LOCATION: {side + ' ' if side else ''}{location}
CONFIDENCE: {confidence}
DESCRIPTION: Possible vascular abnormality detected in this slice. High-intensity region consistent with contrast-enhanced vessel. Further review recommended.

Note: This is a DEMO result for presentation purposes."""
                    
                    findings.append(MedGemmaFinding(
                        slice_index=idx,
                        slice_number=idx + 1,
                        response=response
                    ))
                    
            except Exception as e:
                print(f"Error reading DICOM {dcm_file}: {e}")
                continue
        
        # Limit findings for demo
        findings = findings[:5]  # Max 5 findings for demo
        
        # Generate summary report
        processing_time = time.time() - start_time
        
        if findings:
            report = f"""📊 DEMO ANALYSIS REPORT

Analyzed {len(sample_indices)} slices from series.
⚠️ Potential findings detected in {len(findings)} slice(s).

Note: This is a DEMO result for presentation/development purposes.
For real analysis, a trained detection model should be used."""
        else:
            report = f"""📊 DEMO ANALYSIS REPORT

Analyzed {len(sample_indices)} slices from series.
✅ No obvious abnormalities detected in sampled slices.

Note: This is a DEMO result for presentation/development purposes."""
        
        result = MedGemmaResponse(
            id=analysis_id,
            status="completed",
            series_uid=series_uid,
            report=report,
            slices_analyzed=len(sample_indices),
            has_findings=len(findings) > 0,
            findings=findings,
            processing_time=processing_time,
        )
        
        medgemma_store[analysis_id] = result
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
    
    data_root = Path("C:/Users/Rayan/Desktop/Main Project")
    series_folder = data_root / "series" / series_uid
    
    if not series_folder.exists():
        raise HTTPException(status_code=404, detail="Series not found")
    
    # Get DICOM files
    dcm_files = sorted(glob.glob(str(series_folder / "*.dcm")))
    if not dcm_files:
        dcm_files = sorted([str(f) for f in series_folder.iterdir() if f.is_file()])
    
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
    - MedGemma: intensity-based detection + LLM analysis
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
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled — reload causes deadlocks with heavy model loading
    )

