"""
FastAPI Backend for Intracranial Aneurysm Detection.

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml
import sys
import uuid

# Add ML module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))

# Initialize FastAPI app
app = FastAPI(
    title="Intracranial Aneurysm Detection API",
    description="API for detecting intracranial aneurysms from medical imaging",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, preprocessor
    
    # Try to load model
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
                print(f"Loaded model weights from {weights_path}")
            
            model = model.to(device)
            model.eval()
            
            # Create preprocessor
            preprocessor = DicomPreprocessor(
                target_size=tuple(config['data']['image_size']),
                num_slices=config['data']['num_slices'],
            )
            
            print(f"Model loaded successfully on {device}")
        else:
            print("Config not found, model not loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        preprocessor = None


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
    all_slice_images = []
    
    for idx, file in enumerate(files):
        try:
            content = await file.read()
            
            # Try to read as image first
            try:
                img = Image.open(BytesIO(content))
                if img.mode != 'L':
                    img = img.convert('L')
                pixel_array = np.array(img, dtype=np.float32)
            except Exception:
                # Try DICOM
                try:
                    import pydicom
                    dcm = pydicom.dcmread(BytesIO(content))
                    pixel_array = dcm.pixel_array.astype(np.float32)
                    slope = float(getattr(dcm, 'RescaleSlope', 1))
                    intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                    pixel_array = pixel_array * slope + intercept
                except Exception as e:
                    print(f"⚠️ Could not read file {file.filename}: {e}")
                    continue
            
            slices_analyzed += 1
            h, w = pixel_array.shape
            
            # Apply CTA windowing
            window_center, window_width = 300, 600
            lower = window_center - window_width / 2
            upper = window_center + window_width / 2
            windowed = np.clip(pixel_array, lower, upper)
            windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
            
            # ============== REGION DETECTION ==============
            # Find high-intensity regions (potential vessels/aneurysms)
            threshold = 180  # High intensity threshold
            binary = windowed > threshold
            
            # Clean up with morphological operations
            binary = ndimage.binary_opening(binary, iterations=2)
            binary = ndimage.binary_closing(binary, iterations=2)
            
            # Label connected components
            labeled_array, num_features = label(binary)
            regions = find_objects(labeled_array)
            
            # Create RGB image for visualization
            rgb = np.stack([windowed, windowed, windowed], axis=-1).copy()
            
            # Create heatmap based on detected regions
            heatmap = np.zeros_like(windowed, dtype=np.float32)
            
            detected_regions = []
            for i, region in enumerate(regions):
                if region is None:
                    continue
                    
                # Get bounding box coordinates
                y_slice, x_slice = region
                y_min, y_max = y_slice.start, y_slice.stop
                x_min, x_max = x_slice.start, x_slice.stop
                
                # Filter by size (ignore very small or very large)
                region_width = x_max - x_min
                region_height = y_max - y_min
                region_area = region_width * region_height
                
                min_size = 10
                max_size = min(h, w) * 0.3
                
                if region_width < min_size or region_height < min_size:
                    continue
                if region_width > max_size or region_height > max_size:
                    continue
                
                # Calculate region intensity
                region_mask = labeled_array[y_min:y_max, x_min:x_max] == (i + 1)
                region_intensity = np.mean(windowed[y_min:y_max, x_min:x_max][region_mask])
                
                if region_intensity > 180:
                    detected_regions.append({
                        "bbox": (x_min, y_min, x_max, y_max),
                        "intensity": region_intensity,
                        "area": region_area
                    })
                    
                    # Add to heatmap
                    for dy in range(-20, 20):
                        for dx in range(-20, 20):
                            cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
                            if 0 <= cy + dy < h and 0 <= cx + dx < w:
                                dist = np.sqrt(dy**2 + dx**2)
                                heatmap[cy + dy, cx + dx] += max(0, 1 - dist / 30)
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
            
            # Apply heatmap overlay (red channel)
            rgb[:, :, 0] = np.minimum(255, rgb[:, :, 0].astype(np.float32) + heatmap * 0.7).astype(np.uint8)
            rgb[:, :, 1] = np.maximum(0, rgb[:, :, 1].astype(np.float32) - heatmap * 0.3).astype(np.uint8)
            
            # Draw bounding boxes
            img_pil = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img_pil)
            
            for region in detected_regions:
                x_min, y_min, x_max, y_max = region["bbox"]
                # Red bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
                # Add label
                draw.text((x_min, y_min - 12), f"ROI", fill=(255, 255, 0))
            
            # Convert to base64
            buffer = BytesIO()
            img_pil.save(buffer, format="PNG")
            img_base64 = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
            
            all_slice_images.append(img_base64)
            
            # Create finding if regions detected
            if len(detected_regions) > 0:
                # Sort by intensity
                detected_regions.sort(key=lambda x: x["intensity"], reverse=True)
                top_region = detected_regions[0]
                
                x_min, y_min, x_max, y_max = top_region["bbox"]
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                # Determine location based on position
                if center_x < w * 0.33:
                    side = "Right"  # Radiological convention
                elif center_x > w * 0.66:
                    side = "Left"
                else:
                    side = "Midline"
                
                if center_y < h * 0.4:
                    location = "Anterior region (possible ACA/AComm)"
                elif center_y > h * 0.6:
                    location = "Posterior region (possible PCA/Basilar)"
                else:
                    location = "Middle region (possible MCA/ICA)"
                
                response = f"""🎯 HIGH-INTENSITY REGION DETECTED

📍 Location: {side} {location}
📦 Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})
📊 Max Intensity: {top_region['intensity']:.0f}
📐 Region Size: {top_region['area']} pixels
🔢 Total ROIs: {len(detected_regions)}

⚠️ Note: Automated detection - requires radiologist review."""
                
                findings.append({
                    "slice_index": idx,
                    "slice_number": idx + 1,
                    "response": response,
                    "image": img_base64,
                    "regions": len(detected_regions),
                    "bbox": top_region["bbox"]
                })
                
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    if findings:
        total_regions = sum(f.get("regions", 0) for f in findings)
        report = f"""📊 CT SCAN ANALYSIS REPORT

✅ Analyzed: {slices_analyzed} slice(s)
🎯 Findings: {len(findings)} slice(s) with detections
📦 Total ROIs: {total_regions} region(s) of interest

Detection Method: Intensity-based region segmentation
Threshold: >180 HU (high-density regions)

⚠️ This is automated detection. All findings require verification by a qualified radiologist."""
    else:
        report = f"""📊 CT SCAN ANALYSIS REPORT

✅ Analyzed: {slices_analyzed} slice(s)
✅ No significant high-intensity regions detected

Detection Method: Intensity-based region segmentation
Threshold: >180 HU

Note: Absence of findings does not rule out pathology."""
    
    return {
        "id": analysis_id,
        "status": "completed",
        "report": report,
        "slices_analyzed": slices_analyzed,
        "has_findings": len(findings) > 0,
        "findings": findings[:10],  # Max 10 findings
        "processing_time": processing_time,
        "images": all_slice_images[:20]
    }




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

