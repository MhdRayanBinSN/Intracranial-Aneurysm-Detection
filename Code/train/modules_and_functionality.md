# Modules and Functionality (Updated Implementation)

This document outlines the current architectural modules and their functionalities, reflecting the shift from the initial design (e.g., YOLOv8) to the specialized 3D Deep Learning pipeline currently implemented.

## 1. 🏥 Data Processing Pipeline
*Responsible for ingesting raw medical scans and preparing them for model inference.*

*   **DICOM Ingestor**:
    *   **Functionality**: Parses raw DICOM directories (CTA/MRA sequences).
    *   **Key Logic**: Handles missing metadata, sorts instances by slice location, and constructs 3D volumes.
    *   **Implementation**: `ml/data/preprocessing.py`.
*   **3D Volume Processor**:
    *   **Functionality**: Converts variable-sized DICOM series into standardized 3D tensors.
    *   **Sub-modules**:
        *   **Windowing**: Applies specific intensity windows (Center: 40, Width: 400 for CTA) to highlight intracranial vessels.
        *   **Normalization**: Percentile-based normalization (0-1 range).
        *   **Resampler**: Resizes volumes to fixed dimensions (e.g., 128x128xD) or isotropic spacing (0.5mm).

## 2. 🧠 AI Inference Engine
*The core analytics layer, replacing the previous "Ensemble Prediction Module".*

*   **Module A: Multi-Task Learner (The "Scout")**
    *   **Replacement for**: Separate Detection/Segmentation models.
    *   **Functionality**: Performs two tasks simultaneously using a shared 3D backbone.
        1.  **Vessel Segmentation**: Generates a 3D mask of the intracranial vessels.
        2.  **Global Detection**: Predicts the probability of aneurysm presence in the entire volume.
    *   **Implementation**: `ml/models/multitask_model.py` (ResNet Encoder + U-Net Decoder).

*   **Module B: ROI Refinement & Classification (The "Expert")**
    *   **Replacement for**: Generic Refinement Module.
    *   **Functionality**: Analyzes specific regions of interest (ROIs) identified by the previous stage.
    *   **Key Components**:
        *   **Vessel-Masked Pooling**: Extracts features *only* from vessel regions, ignoring background noise.
        *   **Location-Aware Transformer**: Uses attention mechanisms to model relationships between the 13 distinct anatomical locations (e.g., ICA, MCA, Basilar Tip).
    *   **Implementation**: `ml/models/roi_classifier.py` (Based on 1st Place Solution).

## 3. 🖥️ Backend Services (API Layer)
*Exposes the AI Engine to the user interface.*

*   **Inference Orchestrator**:
    *   **Functionality**: Manages the end-to-end flow: `Upload -> Preprocess -> Multi-Task Inference -> Result Aggregation`.
    *   **Logic**: Handles loading checkpoints (PyTorch `.pth` files) and running inference logic (`inference_pipeline.py`).
*   **FastAPI Gateway**:
    *   **Functionality**: Provides REST Endpoints.
        *   `POST /upload`: Receives DICOM zip files.
        *   `POST /predict`: Triggers analysis and returns JSON results (probabilities, bounding boxes).

## 4. 💻 User Interface (React Dashboard)
*The user-facing application layer.*

*   **Medical Viewer Component**:
    *   **Functionality**: Renders 3D medical slices in the browser. Supports scrolling through z-axis slices.
    *   **Overlay Layer**: Draws bounding boxes and segmentation masks over the raw scans to highlight detected aneurysms.
*   **Report Dashboard**:
    *   **Functionality**: Displays the "Per-Location" probability scores (e.g., "Left ICA: 95%").
    *   **Alert System**: Visual indicators for high-risk findings.

---

### Summary of Changes (Old vs. New)

| Feature | Old Plan (Previous) | **New Implemented Plan** |
| :--- | :--- | :--- |
| **Segmentation** | YOLO v8 (2D Object Detection) | **Multi-Task U-Net (True 3D Segmentation)** |
| **Detection** | Basic 3D CNN | **Location-Aware Transformer + ROI Pooling** |
| **Refinement** | Probability Scoring | **Vessel-Masked Feature Extraction** |
| **Locations** | General Detection | **13 Specific Anatomical Classifications** |
