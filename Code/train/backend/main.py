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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
