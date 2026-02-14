"""
FastAPI Backend for Intracranial Aneurysm Detection.

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

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
import yaml
import sys
import uuid

# Add ML module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))

# Initialize FastAPI app
# app = FastAPI(...) is moved below to include lifespan

# CORS middleware moved to after app initialization

# Global variables for model
model = None
preprocessor = None
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
    global model, preprocessor
    
    # --- STARTUP ---
    print("🚀 Starting MedGemma Backend...")
    try:
        from models import create_model
        from data import DicomPreprocessor
        
        # Load config
        config_path = Path(__file__).parent.parent / 'ml' / 'config' / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Create model
            model = create_model(config)
            
            # Load weights if available
            weights_path = Path(__file__).parent.parent / 'ml' / 'checkpoints' / 'best.pt'
            if weights_path.exists():
                checkpoint = torch.load(weights_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model weights from {weights_path}")
            
            model = model.to(device)
            model.eval()
            
            # Create preprocessor
            preprocessor = DicomPreprocessor(
                target_size=tuple(config['data']['image_size']),
                num_slices=config['data']['num_slices'],
            )
            
            print(f"✅ Model loaded successfully on {device}")
        else:
            print("⚠️ Config not found, model not loaded (Inference Only Mode)")
            
    except Exception as e:
        print(f"⚠️ Error loading legacy model (Non-critical if using MedGemma): {e}")
        model = None
        preprocessor = None
        
    yield
    
    # --- SHUTDOWN ---
    print("🛑 Shutting down MedGemma Backend...")
    # Clean up resources if needed
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
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/medgemma/analyze-upload")
async def analyze_uploaded_files(files: List[UploadFile] = File(...)):
    """
    Analyze uploaded CT slices with bounding box and heatmap visualization.
    
    Uses intensity-based region detection to find high-density areas.
    """
    import time
    import base64
    from io import BytesIO
    from PIL import Image, ImageDraw
    from scipy import ndimage
    from scipy.ndimage import label, find_objects
    
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]
    
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    print(f"📤 Received {len(files)} files for analysis")
    
    findings = []
    slices_analyzed = 0
    
    # Create temp directory for this analysis's images
    analysis_img_dir = os.path.join(tempfile.gettempdir(), "medgemma_images", analysis_id)
    os.makedirs(analysis_img_dir, exist_ok=True)
    
    # 1. Collect all valid slices from inputs (DICOM & NIfTI)
    all_slices = []
    

    for file in files:
        fname = file.filename.lower()
        try:
            content = await file.read()
            
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
                # Handle DICOM
                try:
                    import pydicom
                    dcm = pydicom.dcmread(BytesIO(content))
                    # Rescale
                    arr = dcm.pixel_array.astype(np.float32)
                    slope = float(getattr(dcm, 'RescaleSlope', 1))
                    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                    arr = arr * slope + intercept
                    
                    all_slices.append({
                        "pixel_array": arr,
                        "filename": fname,
                        "original_filename": fname
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

    # 2. Analyze the collected slices
    for idx, slice_data in enumerate(all_slices):
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
            
            # ============ KAGGLE WINNING PREPROCESSING & MULTI-MODALITY ============
            try:
                from data.advanced_preprocessing import KaggleWinningPreprocessor
                preprocessor = KaggleWinningPreprocessor()
                
                # 1. Detect Modality
                modality = preprocessor.detect_modality(pixel_array)
                
                rgb = None
                windowed_for_detection = None
                
                if modality == 'CT':
                    # === CT PIPELINE (Multi-Window) ===
                    # Create 3-channel RGB: Brain, Vessel, Stroke
                    rgb = preprocessor.create_multi_window_image(pixel_array)
                    
                    # Apply CLAHE
                    if preprocessor.use_clahe:
                        for c in range(3):
                            rgb[:, :, c] = preprocessor.apply_clahe(rgb[:, :, c])
                    
                    # Convert to 0-255
                    rgb = (rgb * 255).astype(np.uint8)
                    
                    # Use Vessel channel (Green) for detection
                    windowed_for_detection = rgb[:, :, 1]
                    threshold = 180  # Fixed threshold for CT vessel window
                    
                else:
                    # === MRI PIPELINE (Percentile Normalization) ===
                    # Normalize and create pseudo-color
                    rgb_norm = preprocessor.preprocess_mri(pixel_array)
                    rgb = (rgb_norm * 255).astype(np.uint8)
                    
                    # Use Red channel (Linear) for detection
                    windowed_for_detection = rgb[:, :, 0]
                    # Adaptive threshold for MRI (e.g., top 5% intensity)
                    threshold = np.percentile(windowed_for_detection, 95)
                
                # ============== ANEURYSM-SPECIFIC REGION DETECTION ==============
                # Aneurysms are small (3-25mm), round/saccular, located near the
                # Circle of Willis (central brain), and distinctly brighter than
                # surrounding tissue. We apply strict filters to avoid flagging
                # normal vessels, bone, and noise.
                
                # 1) DYNAMIC THRESHOLD — top 1% intensity (much stricter)
                detection_threshold = np.percentile(windowed_for_detection, 99)
                # Ensure minimum threshold to avoid near-black slices triggering
                detection_threshold = max(detection_threshold, 200 if modality == 'CT' else threshold * 1.1)
                
                binary = windowed_for_detection > detection_threshold
                
                # 2) AGGRESSIVE MORPHOLOGICAL CLEANING — remove noise
                binary = ndimage.binary_opening(binary, iterations=3)
                binary = ndimage.binary_closing(binary, iterations=2)
                
                # Label connected components
                labeled_array, num_features = label(binary)
                regions = find_objects(labeled_array)
                
                # Create heatmap
                heatmap = np.zeros((h, w), dtype=np.float32)
                
                detected_regions = []
                for i, region in enumerate(regions):
                    if region is None:
                        continue
                        
                    y_slice, x_slice = region
                    y_min, y_max = y_slice.start, y_slice.stop
                    x_min, x_max = x_slice.start, x_slice.stop
                    
                    region_width = x_max - x_min
                    region_height = y_max - y_min
                    bbox_area = region_width * region_height
                    
                    # 3) SIZE FILTER — aneurysms are ~3-25mm
                    # At typical CT resolution (~0.5mm/pixel), that's ~6-50 pixels
                    min_size = 15   # ~7mm minimum (smaller = noise/artifacts)
                    max_size = 120  # ~60mm maximum (larger = normal structures)
                    
                    if region_width < min_size or region_height < min_size:
                        continue
                    if region_width > max_size or region_height > max_size:
                        continue
                    
                    # 4) CENTRAL BRAIN FILTER — Circle of Willis is in central region
                    # Exclude regions near edges (skull, scalp, etc.)
                    margin_x = w * 0.15  # 15% margin from each edge
                    margin_y = h * 0.15
                    cx_region = (x_min + x_max) / 2
                    cy_region = (y_min + y_max) / 2
                    
                    if cx_region < margin_x or cx_region > (w - margin_x):
                        continue  # Too close to left/right edge (likely skull)
                    if cy_region < margin_y or cy_region > (h - margin_y):
                        continue  # Too close to top/bottom edge
                    
                    # 5) CIRCULARITY FILTER — aneurysms are round/saccular
                    # Compute actual pixel count vs bounding box area
                    region_mask = labeled_array[y_min:y_max, x_min:x_max] == (i + 1)
                    pixel_count = np.sum(region_mask)
                    
                    # Need minimum fill to not be noise
                    if pixel_count < 50:
                        continue
                    
                    # Circularity = actual_pixels / bbox_area  (1.0 = perfectly fills box)
                    # Aneurysms are round so > 0.35; elongated vessels < 0.2
                    circularity = pixel_count / bbox_area if bbox_area > 0 else 0
                    if circularity < 0.35:
                        continue  # Too elongated (likely a vessel, not aneurysm)
                    
                    # Aspect ratio check — aneurysms are roughly circular
                    aspect_ratio = max(region_width, region_height) / (min(region_width, region_height) + 1e-6)
                    if aspect_ratio > 3.0:
                        continue  # Too stretched — linear vessel, not saccular
                    
                    # 6) INTENSITY CHECK — must be significantly above threshold
                    region_intensity = np.mean(windowed_for_detection[y_min:y_max, x_min:x_max][region_mask])
                    
                    if region_intensity > detection_threshold:
                        detected_regions.append({
                            "bbox": (x_min, y_min, x_max, y_max),
                            "intensity": float(region_intensity),
                            "area": int(pixel_count),
                            "circularity": float(circularity),
                            "aspect_ratio": float(aspect_ratio),
                            "modality": modality
                        })
                        
                        # Add to heatmap (focused Gaussian)
                        cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
                        radius = max(region_width, region_height)
                        for dy in range(-radius, radius+1):
                            for dx in range(-radius, radius+1):
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    dist = np.sqrt(dy**2 + dx**2)
                                    if dist <= radius:
                                        heatmap[ny, nx] += max(0, 1 - dist / radius)
                
                # Apply subtle heatmap overlay only where detections exist
                if heatmap.max() > 0:
                    heatmap = (heatmap / heatmap.max())
                    rgb[:, :, 0] = np.minimum(255, rgb[:, :, 0].astype(np.float32) + heatmap * 150).astype(np.uint8)
                    rgb[:, :, 1] = np.maximum(0, rgb[:, :, 1].astype(np.float32) - heatmap * 50).astype(np.uint8)
            
            except Exception as e:
                 # Fallback if module not found or error
                 print(f"⚠️ Advanced preprocessing error: {e}. Using fallback.")
                 window_center, window_width = 300, 600
                 lower = window_center - window_width / 2
                 upper = window_center + window_width / 2
                 windowed = np.clip(pixel_array, lower, upper)
                 windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
                 rgb = np.stack([windowed]*3, axis=-1)
                 detected_regions = []
            
            # Only process and save images for slices with actual findings
            if len(detected_regions) > 0:
                # Draw bounding boxes on detected slices only
                img_pil = Image.fromarray(rgb)
                draw = ImageDraw.Draw(img_pil)
                
                for region in detected_regions:
                    x_min, y_min, x_max, y_max = region["bbox"]
                    draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
                    draw.text((x_min, y_min - 12), f"ROI", fill=(255, 255, 0))
                
                # Save only finding images to disk
                img_path = os.path.join(analysis_img_dir, f"slice_{idx}.png")
                img_pil.save(img_path, format="PNG")
                img_url = f"/medgemma/finding-image/{analysis_id}/{idx}"
                
                print(f"  >> Slice {idx+1}: {len(detected_regions)} suspicious region(s) found")

                # Sort by intensity
                detected_regions.sort(key=lambda x: x["intensity"], reverse=True)
                top_region = detected_regions[0]
                
                x_min, y_min, x_max, y_max = top_region["bbox"]
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                # Map to one of the 13 anatomical locations based on spatial position
                # Radiological convention: left in image = patient right
                if center_x < w * 0.5:
                    side = "Right"  # Radiological convention
                else:
                    side = "Left"
                
                # Determine arterial zone by vertical + horizontal position
                if center_y < h * 0.3:
                    # Superior / Anterior region
                    if abs(center_x - w * 0.5) < w * 0.15:
                        location_name = "Anterior Communicating Artery"
                    else:
                        location_name = f"{side} Anterior Cerebral Artery"
                elif center_y < h * 0.45:
                    # Supraclinoid ICA / MCA region
                    if abs(center_x - w * 0.5) > w * 0.25:
                        location_name = f"{side} Middle Cerebral Artery"
                    else:
                        location_name = f"{side} Supraclinoid Internal Carotid Artery"
                elif center_y < h * 0.65:
                    # Infraclinoid ICA / PComm region
                    if abs(center_x - w * 0.5) > w * 0.2:
                        location_name = f"{side} Infraclinoid Internal Carotid Artery"
                    else:
                        location_name = f"{side} Posterior Communicating Artery"
                else:
                    # Posterior region
                    if abs(center_x - w * 0.5) < w * 0.1:
                        location_name = "Basilar Tip"
                    else:
                        location_name = "Other Posterior Circulation"
                
                # Generate MedGemma Analysis for the top finding
                try:
                    # Lazy load MedGemma
                    from main import get_medgemma
                    medgemma = get_medgemma()
                    
                    # Analyze the specific region image
                    region_img = Image.fromarray(rgb)
                    
                    # Crop to region context (slightly larger than bbox)
                    pad = 50
                    crop_box = (
                        max(0, top_region["bbox"][0] - pad),
                        max(0, top_region["bbox"][1] - pad),
                        min(w, top_region["bbox"][2] + pad),
                        min(h, top_region["bbox"][3] + pad)
                    )
                    cropped_img = region_img.crop(crop_box)
                    
                    # Run Kaggle-specific analysis
                    slice_info = f"Slice {idx+1}"
                    if hasattr(slice_file, 'filename'):
                        slice_info += f" ({slice_file.filename})"
                    
                    llm_analysis = medgemma.analyze_for_kaggle(cropped_img, modality=modality, slice_info=slice_info)
                    
                    response = f"""HIGH-INTENSITY REGION DETECTED

Location: {location_name}
Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})
Max Intensity: {top_region['intensity']:.0f}

MEDGEMMA ANALYSIS:
{llm_analysis}

Note: Automated detection - requires radiologist review."""
                except Exception as e:
                    print(f"MedGemma analysis failed: {e}")
                    response = f"""HIGH-INTENSITY REGION DETECTED

Location: {location_name}
Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})
Max Intensity: {top_region['intensity']:.0f}

(MedGemma detailed analysis unavailable)"""
                
                findings.append({
                    "slice_index": idx,
                    "slice_number": idx + 1,
                    "response": response,
                    "image": img_url,
                    "regions": len(detected_regions),
                    "bbox": top_region["bbox"],
                    "location": location_name,
                    "intensity": top_region["intensity"]
                })
                
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
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
    
    return {
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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

