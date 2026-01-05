# Intracranial Aneurysm Detection - Project Documentation

## 📋 Project Overview

This project implements a **deep learning pipeline** for detecting and localizing intracranial aneurysms in brain imaging data (CT/MRI scans). The implementation is based on the **RSNA 2025 Kaggle Competition** and incorporates techniques from the **1st place winning solution**.

---

## 🎯 Problem Statement

- **Task**: Detect presence and location of brain aneurysms
- **Input**: 3D brain scans (DICOM format) - CTA, MRA, MRI
- **Output**: 
  - Binary classification: Aneurysm Present (Yes/No)
  - Multi-label classification: 13 anatomical locations

---

## 📊 Dataset

| Metric | Value |
|--------|-------|
| Total Scans | 4,348 |
| With Aneurysm | 1,864 (43%) |
| Without Aneurysm | 2,484 (57%) |
| Segmentation Masks | 178 |
| Modalities | CTA (42%), MRA (29%), MRI T2 (23%), MRI T1post (7%) |

### 13 Anatomical Locations:
1. Left Infraclinoid ICA
2. Right Infraclinoid ICA
3. Left Supraclinoid ICA
4. Right Supraclinoid ICA
5. Left Middle Cerebral Artery
6. Right Middle Cerebral Artery
7. Anterior Communicating Artery
8. Left Anterior Cerebral Artery
9. Right Anterior Cerebral Artery
10. Left Posterior Communicating Artery
11. Right Posterior Communicating Artery
12. Basilar Tip
13. Other Posterior Circulation

---

## 🏗️ Project Structure

```
ml/
├── config/
│   └── config.yaml                 # All configuration parameters
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                  # Main dataset class
│   ├── preprocessing.py            # DICOM preprocessing
│   ├── augmentations.py            # 3D data augmentation
│   └── segmentation_dataset.py     # Multi-task dataset
│
├── models/
│   ├── __init__.py
│   ├── backbone.py                 # 3D ResNet backbones
│   ├── classifier.py               # Classification heads
│   ├── model.py                    # Basic detector model
│   ├── multitask_model.py          # Multi-task U-Net
│   └── roi_classifier.py           # 1st place solution model
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                  # Training loop
│   ├── multitask_trainer.py        # Multi-task training
│   ├── losses.py                   # Loss functions
│   └── metrics.py                  # Evaluation metrics
│
├── preprocessing/
│   ├── __init__.py
│   └── dicom_to_nifti.py           # DICOM to NIfTI conversion
│
├── nnunet/
│   ├── __init__.py
│   └── create_dataset.py           # nnU-Net dataset format
│
├── roi/
│   ├── __init__.py
│   └── roi_extraction.py           # ROI extraction & vessel pooling
│
├── train.py                        # Basic training script
├── train_multitask.py              # Multi-task training script
├── train_roi_classifier.py         # 1st place model training
├── inference.py                    # Basic inference
├── inference_pipeline.py           # Full pipeline inference
├── test_pipeline.py                # Pipeline testing
├── requirements.txt                # Dependencies
└── setup_env.bat                   # Environment setup
```

---

## 🔧 Implementation Details

### Phase 1: Data Pipeline

#### 1.1 DICOM Loading (`data/preprocessing.py`)
- Loads DICOM files from zip archive
- Handles multi-frame DICOMs with missing attributes
- Default parameters for problematic files:
  - Spacing: (0.5, 0.5) mm
  - Slice thickness: 5.0 mm

#### 1.2 Preprocessing
- **CT (CTA)**: Windowing with center=40, width=400
- **MRI (MRA, T1, T2)**: Percentile normalization (1st-99th)
- **Resizing**: All volumes resized to (D, 128, 128) or configurable

#### 1.3 Data Augmentation (`data/augmentations.py`)
- Random 3D flips (all axes)
- Random rotations (±15°)
- Intensity shift/scale
- Gaussian noise
- Gaussian blur

---

### Phase 2: Models

#### 2.1 Basic Classification (`models/model.py`)
- **AneurysmDetector**: 3D ResNet + Multi-label classifier
- Backbone options: ResNet-18, 34, 50
- Output: 14 labels (13 locations + presence)

#### 2.2 Multi-Task Model (`models/multitask_model.py`)
- **MultiTaskUNet**: Combined classification + segmentation
- Shared encoder, separate decoder heads
- Classification via global average pooling
- Segmentation via U-Net decoder

#### 2.3 ROI Classifier (`models/roi_classifier.py`) - 1st Place
- **Location-Aware Transformer**: Models inter-location relationships
- **Vessel-Masked Pooling**: Extracts features per anatomical location
- **Auxiliary Sphere Detection**: Learns aneurysm centers
- Parameters: 21.5 million

---

### Phase 3: Loss Functions (`training/losses.py`)

| Loss | Purpose |
|------|---------|
| `WeightedBCELoss` | Class imbalance handling |
| `FocalLoss` | Hard example mining |
| `AsymmetricLoss` | Asymmetric positive/negative weighting |
| `DiceLoss` | Segmentation overlap |
| `DiceBCELoss` | Combined Dice + BCE |
| `MultiTaskLoss` | Classification + Segmentation combined |
| `AuxiliarySphereLoss` | Focal-Tversky++ for sparse targets |

---

### Phase 4: Training

#### 4.1 Basic Training (`train.py`)
```bash
python train.py --epochs 50 --batch-size 4 --lr 1e-4
```

#### 4.2 Multi-Task Training (`train_multitask.py`)
```bash
python train_multitask.py --epochs 50 --batch-size 2
```

#### 4.3 ROI Classifier Training (`train_roi_classifier.py`)
```bash
python train_roi_classifier.py --epochs 30 --batch-size 2
```

Features:
- Mixed precision (FP16)
- Gradient accumulation
- EMA (Exponential Moving Average) weights
- Cosine annealing with warmup
- Multi-loss training

---

### Phase 5: Inference

#### 5.1 Basic Inference (`inference.py`)
```bash
python inference.py --checkpoint checkpoints/best.pth --series-uid <UID>
```

#### 5.2 Full Pipeline (`inference_pipeline.py`)
```bash
python inference_pipeline.py --checkpoint checkpoints/roi_classifier/best.pth
```

Pipeline stages:
1. Load & preprocess volume
2. Segment vessels (nnU-Net placeholder)
3. Extract ROI
4. Classify locations + presence

---

## 📈 Metrics

### Classification Metrics (`training/metrics.py`)
- **MultilabelAUC**: Weighted average AUC across all labels
- **Per-location AUC**: Individual AUC for each location
- **Competition Metric**: 0.5 × Location AUC + 0.5 × Presence AUC

### Segmentation Metrics
- **Dice Score**: Overlap between predicted and ground truth masks
- **Per-class Dice**: For each vessel class

---

## 🔑 Key Techniques from 1st Place Solution

1. **Coarse-to-Fine Segmentation**
   - Coarse model (1.0mm) → Find candidate region
   - Fine models (0.45mm) → Detailed segmentation

2. **Vessel-Masked Pooling**
   - Use vessel masks to extract location-specific features
   - Each of 13 locations gets dedicated feature vector

3. **Location-Aware Transformer**
   - 8 attention heads, 2 layers
   - Models relationships between vessel locations

4. **Loss Weighting**
   - Location loss: 0.1
   - Presence loss: 0.05
   - Aux sphere loss: 1.0

5. **EMA Weights**
   - Use exponential moving average for inference

---

## 🛠️ Setup Instructions

### 1. Create Environment
```bash
conda create -n aneurysm python=3.10
conda activate aneurysm
```

### 2. Install Dependencies
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install nnunetv2 monai
```

### 3. Configure Paths
Edit `config/config.yaml`:
```yaml
data:
  zip_path: "C:/path/to/rsna-intracranial-aneurysm-detection.zip"
  cache_dir: "./cache"
```

---

## 📁 Files Created/Modified

### Core Files
| File | Purpose |
|------|---------|
| `data/dataset.py` | Main dataset class for loading brain scans |
| `data/preprocessing.py` | DICOM preprocessing with windowing |
| `data/augmentations.py` | 3D data augmentation transforms |
| `data/segmentation_dataset.py` | Multi-task dataset with masks |
| `models/backbone.py` | 3D ResNet backbones |
| `models/model.py` | Basic AneurysmDetector |
| `models/multitask_model.py` | Multi-task U-Net model |
| `models/roi_classifier.py` | 1st place ROI classifier |
| `training/trainer.py` | Training loop implementation |
| `training/losses.py` | All loss functions |
| `training/metrics.py` | Evaluation metrics |

### Additional Modules
| File | Purpose |
|------|---------|
| `preprocessing/dicom_to_nifti.py` | DICOM → NIfTI conversion |
| `nnunet/create_dataset.py` | nnU-Net dataset format |
| `roi/roi_extraction.py` | ROI extraction + vessel pooling |

### Scripts
| File | Purpose |
|------|---------|
| `train.py` | Basic classification training |
| `train_multitask.py` | Multi-task training |
| `train_roi_classifier.py` | 1st place model training |
| `inference.py` | Basic inference |
| `inference_pipeline.py` | Full pipeline inference |
| `test_pipeline.py` | Pipeline testing |

---

## 🚀 How to Run

### Quick Test
```bash
python test_pipeline.py
```

### Train Model
```bash
python train.py --epochs 50 --debug  # Debug mode with 10 samples
```

### Run Inference
```bash
python inference_pipeline.py
```

---

## 📊 Expected Results

| Metric | 1st Place | Our Implementation |
|--------|-----------|-------------------|
| Weighted AUC | 0.916 | TBD (need training) |
| Location AUC | 0.916 | TBD |
| Presence AUC | 0.915 | TBD |

---

## 🔮 Future Work

1. **Full nnU-Net Training**: Train vessel segmentation models
2. **5-Fold Cross-Validation**: Proper validation strategy
3. **Test-Time Augmentation**: Average predictions from augmented inputs
4. **Model Ensemble**: Combine multiple models
5. **TensorRT Optimization**: Faster inference for deployment

---

## 📚 References

1. RSNA 2025 Kaggle Competition: [Link](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
2. 1st Place Solution: [GitHub](https://github.com/uchiyama33/rsna2025_1st_place)
3. nnU-Net: [GitHub](https://github.com/MIC-DKFZ/nnUNet)
4. MONAI: [GitHub](https://github.com/Project-MONAI/MONAI)
