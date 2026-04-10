"""
Dataset Information Router — Modular FastAPI router.
Provides RSNA Intracranial Aneurysm Detection dataset statistics.

Include in main.py with:
    from dataset_router import router as dataset_router
    app.include_router(dataset_router)
"""

from fastapi import APIRouter
from pathlib import Path
import os

router = APIRouter(prefix="/dataset", tags=["Dataset"])

# Dataset paths
DATA_ROOT = Path("C:/Users/Rayan/Desktop/Main Project")
TRAIN_CSV = DATA_ROOT / "train.csv"
ZIP_FILE = DATA_ROOT / "rsna-intracranial-aneurysm-detection.zip"

LOCATION_COLUMNS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]


@router.get("/info")
def get_dataset_info():
    """
    Return full statistics about the RSNA Intracranial Aneurysm Detection dataset.
    Reads train.csv and returns computed statistics for the frontend.
    """
    try:
        import pandas as pd

        if not TRAIN_CSV.exists():
            return {"error": f"train.csv not found at {TRAIN_CSV}", "available": False}

        df = pd.read_csv(TRAIN_CSV)

        # --- Core counts ---
        total_series = len(df)
        positive = int(df["Aneurysm Present"].sum())
        negative = total_series - positive
        positive_pct = round(positive / total_series * 100, 1)
        negative_pct = round(negative / total_series * 100, 1)

        # --- Modality distribution ---
        modality_counts = df["Modality"].value_counts().to_dict()

        # --- Sex distribution ---
        sex_counts = df["PatientSex"].value_counts().to_dict()

        # --- Age statistics ---
        age_mean = round(float(df["PatientAge"].mean()), 1)
        age_min = int(df["PatientAge"].min())
        age_max = int(df["PatientAge"].max())
        age_median = round(float(df["PatientAge"].median()), 1)

        # Age buckets for histogram
        buckets = [
            ("<30", int((df["PatientAge"] < 30).sum())),
            ("30-39", int(((df["PatientAge"] >= 30) & (df["PatientAge"] < 40)).sum())),
            ("40-49", int(((df["PatientAge"] >= 40) & (df["PatientAge"] < 50)).sum())),
            ("50-59", int(((df["PatientAge"] >= 50) & (df["PatientAge"] < 60)).sum())),
            ("60-69", int(((df["PatientAge"] >= 60) & (df["PatientAge"] < 70)).sum())),
            ("70-79", int(((df["PatientAge"] >= 70) & (df["PatientAge"] < 80)).sum())),
            ("80+", int((df["PatientAge"] >= 80).sum())),
        ]

        # --- Location distribution (aneurysm-positive cases per location) ---
        location_counts = {}
        for col in LOCATION_COLUMNS:
            if col in df.columns:
                location_counts[col] = int(df[col].sum())

        # --- Dataset file info ---
        zip_size_gb = None
        if ZIP_FILE.exists():
            zip_size_gb = round(os.path.getsize(ZIP_FILE) / 1e9, 2)

        return {
            "available": True,
            "overview": {
                "total_series": total_series,
                "positive_cases": positive,
                "negative_cases": negative,
                "positive_pct": positive_pct,
                "negative_pct": negative_pct,
                "total_locations": len(LOCATION_COLUMNS),
            },
            "modality": modality_counts,
            "sex": sex_counts,
            "age": {
                "mean": age_mean,
                "median": age_median,
                "min": age_min,
                "max": age_max,
                "buckets": [{"label": b[0], "count": b[1]} for b in buckets],
            },
            "locations": [
                {"name": name, "count": count, "pct": round(count / positive * 100, 1) if positive > 0 else 0}
                for name, count in sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
            ],
            "files": {
                "train_csv": str(TRAIN_CSV),
                "zip_path": str(ZIP_FILE),
                "zip_size_gb": zip_size_gb,
            },
            "competition": {
                "name": "RSNA 2023 Intracranial Aneurysm Detection",
                "host": "Radiological Society of North America (RSNA)",
                "year": 2023,
                "platform": "Kaggle",
                "task": "Multi-label binary classification across 13 anatomical locations",
                "imaging": "CT Angiography (CTA), MR Angiography (MRA), MRI",
                "url": "https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection",
            },
        }

    except Exception as e:
        return {"available": False, "error": str(e)}
