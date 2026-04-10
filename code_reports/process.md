# 🧠 Aneurysm Detection - Simple Explanation

## What Does This Project Do?

**Input:** CT Scan of Brain (DICOM images)  
**Output:** "Aneurysm detected at [location] with [X]% probability"

---

## 🔄 How It Works (5 Steps)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. INPUT   │───▶│ 2. PREPROCESS│───▶│  3. MODEL   │───▶│ 4. PREDICT  │───▶│  5. OUTPUT  │
│  CT Scan    │    │  Normalize   │    │  nnU-Net    │    │  Heatmaps   │    │  Locations  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

---

## Step 1: Input (CT Scan)

- **What:** 200-400 DICOM slice images per patient
- **Size:** 512 × 512 pixels per slice
- **Format:** DICOM (.dcm files)

```
Slice 1    Slice 2    Slice 3    ...    Slice 300
  ↓          ↓          ↓                  ↓
[2D Image] [2D Image] [2D Image]  ...  [2D Image]
              ↓
        Stacked into 3D Volume
```

---

## Step 2: Preprocessing

**Why?** Different CT machines produce different intensity values.

| Step | What It Does | Why? |
|------|-------------|------|
| **Clip** | Limit values: 0-600 HU | Focus on blood vessels |
| **Normalize** | Scale to 0-1 range | Consistent input |
| **Resample** | Fix spacing to 0.5mm | Same resolution |

```
Raw CT: [-1000, 3000] → Clipped: [0, 600] → Normalized: [0.0, 1.0]
```

---

## Step 3: Model (nnU-Net)

### What is nnU-Net?

**"no-new-Net U-Net"** - A self-configuring medical image segmentation framework.

```
                    ENCODER                              DECODER
                 (Compress & Learn)                   (Expand & Locate)
                 
    Input ─────▶ [Conv Block] ─────▶ [Conv Block] ─────▶ [Conv Block] ─────▶ Output
    512×512       ↓                    ↓                    ↑               13 Channels
                256×256              128×128               ↑                (one per
                  ↓                    ↓                   ↑                location)
                128×128   ──────── [Bottleneck] ─────────▶ ↑
                           (Compressed representation)
```

### Key Architecture Features

| Feature | What | Why |
|---------|------|-----|
| **3D Convolutions** | Process volume, not slices | Aneurysms are 3D objects |
| **Residual Connections** | Skip connections | Prevent vanishing gradients |
| **Multi-scale** | Look at different sizes | Detect small & large aneurysms |

---

## Step 4: Prediction (Blob Regression)

### The Innovation: Blob Regression

Instead of marking exact pixels, predict a **"blob" (sphere)** around aneurysm locations.

```
Traditional Segmentation:          Blob Regression:
Mark exact aneurysm pixels         Mark probability sphere

     ████                              ░░░░░
    ██████                            ░█████░
     ████                             ░██████░
                                      ░█████░
                                       ░░░░░
                                       
  (Hard to learn)                  (Easier to learn)
```

### 13 Output Channels (Locations)

| # | Location | # | Location |
|---|----------|---|----------|
| 1 | Right MCA | 8 | Left Infraclinoid ICA |
| 2 | Left MCA | 9 | Right Infraclinoid ICA |
| 3 | Right ACA | 10 | Left P-Comm |
| 4 | Left ACA | 11 | Right P-Comm |
| 5 | Anterior Communicating | 12 | Basilar Tip |
| 6 | Left Supraclinoid ICA | 13 | Other Posterior |
| 7 | Right Supraclinoid ICA | | |

---

## Step 5: Output

```python
# Example Output
{
    "aneurysm_detected": True,
    "probability": 0.742,
    "location": "Right Middle Cerebral Artery",
    "risk_level": "HIGH"
}
```

---

## 🏗️ Model Architecture Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    ResidualEncoderUNet                         │
├────────────────────────────────────────────────────────────────┤
│  Input: 3D CT Volume (D × H × W)                               │
│                                                                │
│  Encoder (6 stages):                                           │
│    Stage 1: 32 filters   → Stage 2: 64 filters                │
│    Stage 3: 128 filters  → Stage 4: 256 filters               │
│    Stage 5: 320 filters  → Stage 6: 320 filters (bottleneck)  │
│                                                                │
│  Decoder (5 stages):                                           │
│    Upsample + Skip Connections from Encoder                   │
│                                                                │
│  Output: 13 probability maps (one per location)               │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Training Details

| Parameter | Value |
|-----------|-------|
| **Epochs** | 1500 |
| **Batch Size** | 32 |
| **Optimizer** | SGD (momentum=0.99) |
| **Learning Rate** | 0.01 → 0.0 (polynomial decay) |
| **Loss Function** | TopK BCE (top 20% hardest samples) |
| **GPU** | NVIDIA RTX 3090 / A100 |

### Loss Function: TopK BCE

```
Standard BCE: Average loss over ALL pixels
TopK BCE:     Average loss over TOP 20% HARDEST pixels

Why? Focus learning on difficult cases, not easy background pixels.
```

---

## 📊 Evaluation Metrics

| Metric | Formula | Our Result |
|--------|---------|------------|
| **Accuracy** | (TP+TN) / Total | 73.3% |
| **Precision** | TP / (TP+FP) | 66.7% |
| **Recall** | TP / (TP+FN) | 66.7% |
| **F1 Score** | 2×P×R / (P+R) | 0.667 |

```
Confusion Matrix:
                  Predicted
               No      Yes
         ┌────────┬────────┐
Actual No│   7    │   2    │ ← 2 False Alarms
         ├────────┼────────┤
     Yes │   2    │   4    │ ← 2 Misses
         └────────┴────────┘
```

---

## 🚀 How to Run

### Option 1: Command Line (Simple)

```bash
# Activate environment
conda activate pretrained_detect

# Run detection
python detect_aneurysm.py -i "path/to/dicom/folder" -o "output.json"
```

### Option 2: Web Interface

```bash
# Terminal 1: Start backend
cd backend
python -m uvicorn main:app --port 8001

# Terminal 2: Start frontend
cd frontend  
npm run dev
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `detect_aneurysm.py` | Standalone CLI detection |
| `evaluate_model.py` | Batch evaluation script |
| `backend/inference.py` | Core inference logic |
| `sol/nnunetv2/` | nnU-Net framework code |
| `checkpoint_epoch_1500.pth` | Trained model weights |

---

## 🔑 Key Concepts to Explain

### 1. Why U-Net?
- Designed for medical images
- Preserves spatial information via skip connections
- Good for localization tasks

### 2. Why 3D?
- Aneurysms are 3D structures
- 2D slices miss context between slices
- Better accuracy with volume analysis

### 3. Why Blob Regression?
- Easier to learn than exact segmentation
- More robust to annotation variations
- Provides smooth probability outputs

### 4. Why TopK Loss?
- Medical images are 99% background
- Standard loss dominated by easy negatives
- TopK focuses on hard examples

---

## 🎓 Simple Analogy

```
Finding aneurysm in CT scan is like:

Finding Waldo in a "Where's Waldo?" book...
  - But the book is 300 pages (slices)
  - Waldo is 3-5mm (tiny)
  - You have 13 different Waldos to find (locations)
  - Some pages don't have Waldo at all

The model learned to:
  1. Quickly skip pages without Waldo (no aneurysm)
  2. Zoom in on suspicious areas
  3. Confirm if it's really Waldo (aneurysm)
  4. Tell you which Waldo it is (location)
```

---

## 📚 References

1. **nnU-Net Paper:** Isensee et al., "nnU-Net: Self-adapting Framework" (2021)
2. **RSNA Competition:** Kaggle RSNA 2024 Lumbar Spine Challenge
3. **MIC-DKFZ Solution:** 7th place Kaggle solution

---

*For detailed technical documentation, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)*
