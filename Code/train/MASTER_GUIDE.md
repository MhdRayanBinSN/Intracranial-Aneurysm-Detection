# Intracranial Aneurysm Detection - Master Project Guide

## 🚀 Introduction
This project implements a state-of-the-art deep learning system for detecting and localizing intracranial aneurysms from 3D medical scans (CTA/MRA). It leverages a **Multi-Task U-Net** for simultaneous vessel segmentation and detection, refined by a **Location-Aware Transformer** (ROI Classifier).

## 📚 Key Documentation
*   **[Project Schedule](project_schedule.md)**: Detailed 10-phase timeline from Dec 8 to April.
*   **[Modules & Functionality](modules_and_functionality.md)**: Technical architecture explaining the shift to 3D Deep Learning.
*   **[Technical Docs](ml/PROJECT_DOCUMENTATION.md)**: In-depth ML implementation details.

---

## 🏗️ System Architecture

### 1. Data Layer (`ml/data/`)
*   **Input**: Raw DICOM directories (CTA, MRA).
*   **Processing**:
    *   **Windowing**: Center 40, Width 400 (CTA).
    *   **Normalization**: Percentile-based (0-1).
    *   **Output**: 3D PyTorch Tensors (Channel, Depth, Height, Width).

### 2. AI Engine (`ml/models/`)
*   **Stage 1: Multi-Task Scout** (`multitask_model.py`)
    *   **Backbone**: 3D ResNet-18/34.
    *   **Heads**: Detection (Classification) + Vessel Segmentation (U-Net).
*   **Stage 2: ROI Expert** (`roi_classifier.py`)
    *   **Input**: Features from vessel regions.
    *   **Logic**: Transformer models 13 specific locations (e.g., ICA, MCA).

### 3. Application Layer
*   **Backend**: Python FastAPI (Inference Orchestrator).
*   **Frontend**: React + Tailwind CSS (Medical Viewer Dashboard).

---

## 🛠️ Setup & Execution

### 1. Environment Setup
```bash
# Create Conda Environment
conda create -n aneurysm python=3.10
conda activate aneurysm

# Install Dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r ml/requirements.txt
```

### 2. Running Training
**Baseline Training:**
```bash
cd ml
python train.py --epochs 50 --batch-size 4
```

**Multi-Task Training (Current Phase):**
```bash
cd ml
python train_multitask.py --epochs 50 --batch-size 2
```

### 3. Running Inference
**Quick Test Pipeline:**
```bash
cd ml
python test_pipeline.py
```

**Full Inference on New Scan:**
```bash
python ml/inference_pipeline.py --scan_path "path/to/dicom_dir"
```

---

## 📂 Project Structure
```text
/
├── ml/                     # Machine Learning Core
│   ├── config/             # Hyperparameters
│   ├── data/               # Preprocessing & Datasets
│   ├── models/             # PyTorch Architectures
│   └── training/           # Training Loops & Losses
├── backend/                # FastAPI Server (Coming Soon)
├── frontend/               # React Dashboard (Coming Soon)
├── project_schedule.md     # Timeline
├── modules_and_functionality.md # Functionality Specs
└── MASTER_GUIDE.md         # This File
```
