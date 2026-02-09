"""
MedGemma + DL Model Web Dashboard
FastAPI backend for medical image analysis
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path
import sys
import os
import uuid
import shutil
import tempfile
from typing import Optional, List, Dict
import json
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASET_ZIP, OUTPUT_DIR, HF_TOKEN

# App setup
app = FastAPI(title="MedGemma Dashboard", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Store for analysis jobs
analysis_jobs: Dict[str, Dict] = {}

# Lazy load MedGemma (heavy model)
medgemma_model = None

def get_medgemma():
    """Lazy load MedGemma model."""
    global medgemma_model
    if medgemma_model is None:
        from inference import MedGemmaInference
        medgemma_model = MedGemmaInference()
    return medgemma_model


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/patients")
async def list_patients():
    """List patients from the dataset."""
    import zipfile
    import pandas as pd
    
    try:
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
            with zf.open('train.csv') as f:
                df = pd.read_csv(f)
        
        # Get unique patients with their modality
        patients = df[['SeriesInstanceUID', 'Modality']].drop_duplicates().head(50)
        
        return {
            "success": True,
            "patients": [
                {
                    "series_uid": row['SeriesInstanceUID'],
                    "modality": row['Modality'],
                    "display_name": f"{row['Modality']} - {row['SeriesInstanceUID'][:20]}..."
                }
                for _, row in patients.iterrows()
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/analyze")
async def analyze_patient(request: Request, background_tasks: BackgroundTasks):
    """Start analysis for a patient."""
    try:
        data = await request.json()
        series_uid = data.get("series_uid")
        model_type = data.get("model_type", "both")  # "medgemma", "resnet", "both"
        
        if not series_uid:
            return {"success": False, "error": "No series_uid provided"}
        
        # Create job ID
        job_id = str(uuid.uuid4())[:8]
        
        # Initialize job
        analysis_jobs[job_id] = {
            "status": "starting",
            "series_uid": series_uid,
            "model_type": model_type,
            "progress": 0,
            "medgemma_results": None,
            "resnet_results": None,
            "started_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        # Run analysis in background
        background_tasks.add_task(run_analysis, job_id, series_uid, model_type)
        
        return {"success": True, "job_id": job_id}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_analysis(job_id: str, series_uid: str, model_type: str):
    """Background task to run analysis."""
    import zipfile
    import io
    import pydicom
    import numpy as np
    
    try:
        analysis_jobs[job_id]["status"] = "loading"
        
        # Load slices from ZIP
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
            series_path = f"series/{series_uid}/"
            dcm_files = sorted([f for f in zf.namelist() 
                               if f.startswith(series_path) and not f.endswith('/')])
            
            if not dcm_files:
                analysis_jobs[job_id]["status"] = "error"
                analysis_jobs[job_id]["error"] = "No DICOM files found"
                return
            
            analysis_jobs[job_id]["total_slices"] = len(dcm_files)
            analysis_jobs[job_id]["status"] = "analyzing"
            
            # MedGemma Analysis (sample slices)
            if model_type in ["medgemma", "both"]:
                try:
                    model = get_medgemma()
                    medgemma_findings = []
                    
                    # Analyze every 20th slice for speed
                    sample_indices = list(range(0, len(dcm_files), 20))
                    
                    for i, idx in enumerate(sample_indices):
                        dcm_file = dcm_files[idx]
                        analysis_jobs[job_id]["progress"] = int((i + 1) / len(sample_indices) * 100)
                        
                        with zf.open(dcm_file) as f:
                            dcm = pydicom.dcmread(io.BytesIO(f.read()))
                            pixel_array = dcm.pixel_array.astype(np.float32)
                            slope = float(getattr(dcm, 'RescaleSlope', 1))
                            intercept = float(getattr(dcm, 'RescaleIntercept', 0))
                            pixel_array = pixel_array * slope + intercept
                        
                        # Quick analysis
                        response = model.analyze_image(
                            pixel_array,
                            prompt="Is there an aneurysm in this brain CTA? Answer: YES or NO, and location if yes.",
                            system_prompt="Be brief. Max 30 words."
                        )
                        
                        has_finding = "yes" in response.lower() and "no" not in response.lower()[:10]
                        
                        if has_finding:
                            medgemma_findings.append({
                                "slice_index": idx,
                                "slice_number": idx + 1,
                                "response": response[:200]
                            })
                    
                    analysis_jobs[job_id]["medgemma_results"] = {
                        "analyzed_slices": len(sample_indices),
                        "findings": medgemma_findings,
                        "has_aneurysm": len(medgemma_findings) > 0
                    }
                except Exception as e:
                    analysis_jobs[job_id]["medgemma_results"] = {"error": str(e)}
            
            # ResNet Analysis (mock for now - would need actual model)
            if model_type in ["resnet", "both"]:
                # Mock ResNet results - replace with actual inference
                analysis_jobs[job_id]["resnet_results"] = {
                    "prediction": "Aneurysm Detected",
                    "confidence": 0.87,
                    "locations": [
                        {"name": "Left MCA", "confidence": 0.92},
                        {"name": "Right ICA", "confidence": 0.78}
                    ],
                    "note": "Mock result - integrate actual 3D ResNet here"
                }
        
        analysis_jobs[job_id]["status"] = "completed"
        analysis_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        analysis_jobs[job_id]["status"] = "error"
        analysis_jobs[job_id]["error"] = str(e)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get analysis job status."""
    if job_id not in analysis_jobs:
        return {"success": False, "error": "Job not found"}
    
    return {"success": True, "job": analysis_jobs[job_id]}


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results."""
    if job_id not in analysis_jobs:
        return {"success": False, "error": "Job not found"}
    
    job = analysis_jobs[job_id]
    
    return {
        "success": True,
        "job_id": job_id,
        "status": job["status"],
        "medgemma": job.get("medgemma_results"),
        "resnet": job.get("resnet_results")
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🏥 MedGemma + DL Model Dashboard")
    print("=" * 60)
    print("🌐 Open: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
