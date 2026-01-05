# Intracranial Aneurysm Detection Project

A complete ML pipeline with React web interface for detecting and classifying intracranial aneurysms from medical imaging (CTA, MRA, MRI).

## 🏗️ Project Structure

```
Code/
├── frontend/          # React + TailwindCSS web interface
├── backend/           # FastAPI server
├── ml/                # ML pipeline (data, models, training)
├── Eda.ipynb          # Exploratory Data Analysis
└── Visualize.ipynb    # DICOM Visualization
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA 11.8+ (for GPU training)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Training
```bash
cd ml
python train.py --config config/config.yaml
```

## 📊 Competition Details

Based on [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)

- **Task**: Multi-label classification (13 anatomical locations + overall presence)
- **Data**: DICOM images (CTA, MRA, MRI)
- **Metric**: Weighted Multilabel AUC

## 🏆 Approach

Inspired by 1st place solution:
1. **Preprocessing**: DICOM to NIfTI conversion
2. **Vessel Segmentation**: nnU-Net for ROI extraction
3. **Classification**: 3D CNN on segmented regions

## 🖥️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | React 18 + TailwindCSS |
| Backend | FastAPI + PyTorch |
| ML | MONAI + PyTorch |
| DICOM Viewer | Cornerstone.js |
