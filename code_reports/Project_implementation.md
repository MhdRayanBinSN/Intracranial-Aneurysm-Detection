# RSNA Intracranial Aneurysm Detection System
## Comprehensive Project Documentation

---

## Table of Contents

### Beginner's Guide (Start Here!)
0. [Prerequisites: Understanding the Basics](#0-prerequisites-understanding-the-basics-for-beginners)
   - 0.1 [What is Deep Learning?](#01-what-is-deep-learning)
   - 0.2 [What is a Neural Network?](#02-what-is-a-neural-network)
   - 0.3 [What is a CNN?](#03-what-is-a-convolutional-neural-network-cnn)
   - 0.4 [What is U-Net?](#04-what-is-u-net-the-base-architecture)
   - 0.5 [What is Residual Network (ResNet)?](#05-what-is-a-residual-network-resnet)
   - 0.6 [What is 3D Convolution?](#06-what-is-3d-convolution)
   - 0.7 [What is nnU-Net?](#07-what-is-nnu-net)
   - 0.8 [Segmentation vs Classification vs Detection](#08-what-is-segmentation-vs-classification-vs-detection)
   - 0.9 [What is Blob Regression?](#09-what-is-blob-regression-key-innovation)
   - 0.10 [What is Loss Function?](#010-what-is-loss-function)
   - 0.11 [What is Backpropagation?](#011-what-is-backpropagation)
   - 0.12 [What is Inference?](#012-what-is-inference)
   - 0.13 [What is Sliding Window?](#013-what-is-sliding-window-inference)
   - 0.14 [What is GPU/CUDA?](#014-what-is-gpucuda)
   - 0.15 [Summary: How Everything Connects](#015-summary-how-everything-connects)
   - 0.16 [Key Terms Glossary](#016-key-terms-glossary)

### Project Details
1. [Problem Statement](#1-problem-statement)
2. [Medical Background](#2-medical-background)
3. [Dataset Overview](#3-dataset-overview)
4. [Solution Architecture](#4-solution-architecture)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Model Architecture](#6-model-architecture)
7. [Training Process](#7-training-process)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Application Architecture](#9-application-architecture)
10. [Results & Metrics](#10-results--metrics)
11. [Challenges Faced](#11-challenges-faced)
12. [Key Learnings](#12-key-learnings)
13. [How to Run](#13-how-to-run)
14. [References](#14-references)

---

## 0. Prerequisites: Understanding the Basics (For Beginners)

Before diving into the project, let's understand the fundamental concepts in simple terms.

### 0.1 What is Deep Learning?

**Simple Explanation:**
Deep Learning is teaching computers to recognize patterns, similar to how our brain works.

```
Traditional Programming:              Deep Learning:
────────────────────────              ──────────────
Input + RULES = Output               Input + Output = RULES (learned)

Example:                              Example:
"If pixel > 200, it's white"         Show 1000 cat pictures,
"If edge detected, it's boundary"    computer learns what a cat looks like
```

**Analogy:** 
- Traditional programming = Following a recipe exactly
- Deep Learning = Learning to cook by tasting many dishes

---

### 0.2 What is a Neural Network?

A neural network is inspired by the human brain. It has **layers of neurons** that process information.

```
        INPUT LAYER          HIDDEN LAYERS           OUTPUT LAYER
        (What we see)        (Processing)            (Decision)
        
          ○ ────────┐
                    ├───── ○ ─────┐
          ○ ────────┤             ├───── ○ ─────┐
                    ├───── ○ ─────┤             ├───── ○  (Aneurysm: Yes/No)
          ○ ────────┤             ├───── ○ ─────┘
                    ├───── ○ ─────┘
          ○ ────────┘
          
       (Image pixels)    (Learn features)     (Make prediction)
```

**How it learns:**
1. **Forward Pass:** Data flows through network, makes a prediction
2. **Check Error:** Compare prediction with correct answer
3. **Backward Pass (Backpropagation):** Adjust weights to reduce error
4. **Repeat:** Do this millions of times until accurate

---

### 0.3 What is a Convolutional Neural Network (CNN)?

CNNs are special neural networks designed for **images**. They use **filters/kernels** to detect patterns.

```
IMAGE PROCESSING WITH CNN:

Original Image          Filter (3x3)           Feature Map
┌─────────────────┐     ┌─────────┐           ┌─────────────┐
│ 0 1 1 0 0 │           │ 1 0 1 │             │ 4 3 4 │
│ 0 0 1 1 0 │     *     │ 0 1 0 │     =       │ 2 4 3 │
│ 0 1 1 0 0 │           │ 1 0 1 │             │ 4 3 4 │
│ 0 0 1 1 0 │           └─────────┘           └─────────────┘
│ 0 1 1 0 0 │           Edge Detector         Edges Found!
└─────────────────┘
```

**What each layer learns:**
- **Layer 1:** Edges, lines, simple shapes
- **Layer 2:** Corners, textures, curves
- **Layer 3:** Parts (eyes, wheels, branches)
- **Layer 4+:** Complete objects (faces, cars, aneurysms)

**Analogy:** 
Like how you recognize a friend:
1. First see edges (outline of face)
2. Then see features (eyes, nose)
3. Finally recognize the person

---

### 0.4 What is U-Net? (The Base Architecture)

**U-Net** is a special CNN architecture shaped like the letter "U", designed for **segmentation** (marking regions in images).

```
                            U-NET ARCHITECTURE
                            
    ENCODER (Downsampling)              DECODER (Upsampling)
    Compress & Learn Features           Restore Size & Details
    
    Input Image ──────────────────────────────────────▶ Output Mask
         │                                                  ▲
         ▼                                                  │
    ┌─────────┐                                       ┌─────────┐
    │  64×64  │─────────── Skip Connection ──────────▶│  64×64  │
    │ features│                                       │ features│
    └────┬────┘                                       └────▲────┘
         │ ↓ Downsample (shrink)              Upsample ↑   │
         ▼                                                  │
    ┌─────────┐                                       ┌─────────┐
    │  32×32  │─────────── Skip Connection ──────────▶│  32×32  │
    │ features│                                       │ features│
    └────┬────┘                                       └────▲────┘
         │ ↓                                          ↑    │
         ▼                                                  │
    ┌─────────┐                                       ┌─────────┐
    │  16×16  │─────────── Skip Connection ──────────▶│  16×16  │
    │ features│                                       │ features│
    └────┬────┘                                       └────▲────┘
         │                                                  │
         └──────────────▶ BOTTLENECK ──────────────────────┘
                         (Deepest learning)
```

**Why "U" shape?**
1. **Encoder (Left side):** Shrinks image, learns "WHAT" is in the image
2. **Bottleneck (Bottom):** Most compressed, deepest understanding
3. **Decoder (Right side):** Expands back, learns "WHERE" things are
4. **Skip Connections (Arrows):** Preserve fine details from encoder to decoder

**Analogy:**
- Encoder = Zooming out to see the big picture
- Decoder = Zooming back in to mark exact locations
- Skip connections = Remembering the details while zoomed out

---

### 0.5 What is a Residual Network (ResNet)?

**Problem:** Very deep networks are hard to train (gradients vanish).

**Solution:** **Residual Connections** - let information skip layers directly.

```
NORMAL NETWORK:                      RESIDUAL NETWORK:
                                     
Input ──▶ Layer 1 ──▶ Layer 2 ──▶ Output     Input ──┬──▶ Layer 1 ──▶ Layer 2 ──┬──▶ Output
                                                     │                          │
                                                     └────── SKIP ──────────────┘
                                                     (Add input directly to output)
```

**Mathematical View:**
```
Normal:    Output = F(input)           # Must learn everything
Residual:  Output = F(input) + input   # Only learn the DIFFERENCE
```

**Analogy:**
- Normal: Write an entire essay from scratch
- Residual: Start with a draft and only write the corrections

**Why it works:**
- Easier to learn "what to change" than "everything from scratch"
- Gradients flow directly through skip connections
- Can train networks with 100+ layers

---

### 0.6 What is 3D Convolution?

Normal CNNs work on 2D images. **3D CNNs** work on volumes (like CT scans with many slices).

```
2D CONVOLUTION (Images):             3D CONVOLUTION (Volumes):

    ┌───────────┐                        ┌───────────┐
    │  Height   │                       /│  Height   │
    │    ×      │                      / │    ×      │
    │  Width    │                     /  │  Width    │
    └───────────┘                    /   │    ×      │
       2D Image                     │   Depth       │
     (1 slice)                      └───────────────┘
                                      3D Volume
                                    (many slices)
                                    
Filter: 3×3                         Filter: 3×3×3
(looks at 9 pixels)                 (looks at 27 voxels)
```

**Why 3D for medical imaging?**
- CT/MRI scans are 3D volumes (stack of 2D slices)
- Aneurysms are 3D structures
- 3D convolutions understand spatial relationships across slices

**Voxel = 3D Pixel:**
```
2D: Pixel (Picture Element)  →  A single colored square
3D: Voxel (Volume Element)   →  A single colored cube
```

---

### 0.7 What is nnU-Net?

**nnU-Net** = "no-new-Net" (because it uses existing ideas smartly, not new architecture)

It's a **self-configuring framework** that automatically determines the best settings for medical image segmentation.

```
TRADITIONAL APPROACH:                    nnU-Net APPROACH:
                                         
You manually decide:                     nnU-Net automatically determines:
- Network architecture                   - Best architecture for YOUR data
- Patch size                            - Optimal patch size
- Batch size                            - Best batch size
- Preprocessing                         - Correct preprocessing
- Augmentation                          - Appropriate augmentation
                                        
Result: Trial and error                 Result: Near-optimal out of the box
        (weeks of experiments)                  (just provide data)
```

**Why nnU-Net is revolutionary:**
1. **Analyzes your dataset** (image sizes, spacing, modality)
2. **Automatically configures** all hyperparameters
3. **Achieves state-of-the-art** on most medical imaging tasks
4. **Works out-of-the-box** without manual tuning

**Analogy:**
- Traditional: Buying ingredients and figuring out how to cook
- nnU-Net: A smart kitchen that looks at your ingredients and cooks the perfect meal automatically

---

### 0.8 What is Segmentation vs Classification vs Detection?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPUTER VISION TASKS COMPARISON                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CLASSIFICATION              DETECTION                SEGMENTATION         │
│  "What is in image?"         "Where are objects?"     "Mark every pixel"   │
│                                                                             │
│  ┌─────────────────┐        ┌─────────────────┐      ┌─────────────────┐   │
│  │                 │        │    ┌─────┐      │      │    ████████     │   │
│  │       🐱        │        │    │ 🐱  │      │      │   ██████████    │   │
│  │                 │        │    └─────┘      │      │   ██████████    │   │
│  │                 │        │                 │      │    ████████     │   │
│  └─────────────────┘        └─────────────────┘      └─────────────────┘   │
│                                                                             │
│  Output: "Cat" (label)      Output: Box coordinates   Output: Mask of cat  │
│  Confidence: 0.95           [x, y, width, height]     (pixel-by-pixel)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Our Project:** Uses **Semantic Segmentation** (but outputs heatmaps, not masks)

---

### 0.9 What is Blob Regression? (Key Innovation!)

Instead of drawing boxes around aneurysms, we predict **heatmaps** where the "hottest" point indicates the aneurysm location.

```
TRADITIONAL DETECTION:                   BLOB REGRESSION:
(Predict bounding boxes)                 (Predict heat maps)

┌─────────────────────────┐             ┌─────────────────────────┐
│                         │             │                         │
│    ┌─────────┐          │             │       ░░░░░░░           │
│    │Aneurysm │          │             │      ░░████░░           │
│    │  Box    │          │             │     ░░██████░░          │
│    └─────────┘          │             │      ░░████░░           │
│                         │             │       ░░░░░░░           │
└─────────────────────────┘             └─────────────────────────┘
   Output: [x, y, w, h]                    Output: Probability at each voxel
   "Box around aneurysm"                   "How likely is aneurysm here?"
```

**What is a "Blob"?**
A blob is a **smooth spherical region** with highest value at center, fading to zero at edges.

```
BLOB VISUALIZATION (2D Cross-section):

    Profile View:                    Top View:
    
    1.0 │      ████                     ░░░░░░░░░░░
        │     ██████                   ░░░░░░░░░░░░░
    0.5 │    ████████                 ░░░░░████░░░░░
        │   ██████████               ░░░░████████░░░░
    0.0 │──────────────              ░░░░░████░░░░░░
        └──────────────               ░░░░░░░░░░░░░
         Center→Edge                 Hottest at center
```

**EDT (Euclidean Distance Transform):**
- Creates a smooth gradient from center (1.0) to edge (0.0)
- The distance from center determines the value
- Results in natural, smooth blobs

**Why Blob Regression works better for aneurysms:**

| Aspect | Bounding Box | Blob Regression |
|--------|--------------|-----------------|
| Shape | Rectangle (poor fit for round aneurysms) | Sphere (perfect for round) |
| Size | Hard to define box size | Naturally adapts to any size |
| Location | Center can be ambiguous | Hottest point = precise location |
| Overlapping | Boxes can overlap confusingly | Heatmaps combine naturally |

---

### 0.10 What is Loss Function?

**Loss Function** = A way to measure "how wrong" the model's predictions are.

```
Prediction: 0.3 (model thinks 30% chance of aneurysm)
Truth: 1.0 (aneurysm actually exists)

Loss = How far off is 0.3 from 1.0?

Lower Loss = Better Model
```

**Binary Cross-Entropy (BCE) Loss:**
The loss function used in our project.

```python
# Simplified formula
BCE_Loss = -[y * log(p) + (1-y) * log(1-p)]

Where:
- y = true label (0 or 1)
- p = predicted probability (0 to 1)
```

**Visual Example:**
```
If true answer is 1 (aneurysm present):

Prediction    Loss
0.99    ──── 0.01  (Very good! Low loss)
0.70    ──── 0.36  (Okay)
0.50    ──── 0.69  (Uncertain)
0.30    ──── 1.20  (Bad)
0.01    ──── 4.61  (Very bad! High loss)
```

**TopK BCE Loss (Our Project):**
Only compute loss on the **top 20% worst predictions**.

```
Why TopK?
- Most voxels are easy (clearly background)
- Only 20% are challenging (near aneurysm edges)
- Focus learning on difficult cases
- Prevents easy cases from dominating
```

---

### 0.11 What is Backpropagation?

**Backpropagation** = The algorithm that teaches neural networks by adjusting weights based on errors.

```
FORWARD PASS:                                BACKWARD PASS:
─────────────                                ──────────────

Input ──▶ Layer1 ──▶ Layer2 ──▶ Output      Error ◀── Layer2 ◀── Layer1 ◀── Input
                                    │         │
                                    ▼         │
                              Calculate ──────┘
                               Error
                               
"Make prediction"                            "Adjust weights to reduce error"
```

**Analogy:**
1. Throw a dart at target (forward pass)
2. See how far you missed (calculate error)
3. Adjust your aim for next throw (backpropagation)
4. Repeat until you hit bullseye

---

### 0.12 What is Inference?

**Training** = Teaching the model (slow, uses lots of data)
**Inference** = Using the trained model to make predictions (fast, on new data)

```
TRAINING PHASE:                          INFERENCE PHASE:
(Happens once, takes days)               (Happens every time, takes seconds)

Thousands of CT scans ──▶ Model ──▶ Trained Weights
                                         │
                                         ▼
                         New CT scan ──▶ Trained Model ──▶ "Aneurysm detected!"
```

---

### 0.13 What is Sliding Window Inference?

CT scans are **too large** to process at once. We process them in **patches** (smaller pieces).

```
SLIDING WINDOW CONCEPT:

Full 3D Volume (too large):           Process in Patches:
┌─────────────────────────┐           
│                         │           ┌───┐
│                         │           │ 1 │ ──▶ Model ──▶ Prediction 1
│      256×256×256        │           └───┘
│        voxels           │           
│                         │           ┌───┐
│                         │           │ 2 │ ──▶ Model ──▶ Prediction 2
└─────────────────────────┘           └───┘
                                      
                                      (slide window and repeat)
```

**50% Overlap:**
```
┌─────────────────────────────────────┐
│Patch 1    │Overlap│    Patch 2     │
│           │       │                │
│   ████████│███████│████████        │
│           │███████│                │
└─────────────────────────────────────┘

Overlap ensures no aneurysm is missed at patch boundaries
```

---

### 0.14 What is GPU/CUDA?

**GPU** (Graphics Processing Unit) = Specialized processor for parallel computations.
**CUDA** = NVIDIA's platform to use GPUs for deep learning.

```
CPU vs GPU:

CPU (Central Processing Unit):        GPU (Graphics Processing Unit):
┌─────────────────────┐               ┌─────────────────────────────────┐
│ ██ │ ██ │ ██ │ ██  │               │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                     │               │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│  4-8 powerful cores │               │ 1000s of small cores           │
│                     │               │                                │
│  Good for complex   │               │ Good for simple parallel       │
│  sequential tasks   │               │ tasks (like image processing)  │
└─────────────────────┘               └─────────────────────────────────┘

Deep Learning:                        
- Millions of simple math operations (multiply, add)
- All can run in parallel ──▶ GPU is 10-100x faster!
```

---

### 0.15 Summary: How Everything Connects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               COMPLETE CONCEPTUAL FLOW OF OUR PROJECT                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CT SCAN (DICOM files)                                                     │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ PREPROCESSING                                                     │      │
│  │ • Stack 2D slices into 3D volume                                 │      │
│  │ • Crop to Region of Interest (200×160×160 mm)                    │      │
│  │ • Normalize intensities (Z-score)                                │      │
│  │ • Resample to standard spacing                                   │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ 3D RESIDUAL ENCODER U-NET (nnU-Net)                              │      │
│  │                                                                   │      │
│  │  Input: 3D Volume (1 channel)                                    │      │
│  │      │                                                           │      │
│  │      ▼                                                           │      │
│  │  ENCODER: 3D Convolutions + Residual Connections                 │      │
│  │  (Learns what aneurysms look like)                              │      │
│  │      │                                                           │      │
│  │      ▼                                                           │      │
│  │  DECODER: Upsampling + Skip Connections                          │      │
│  │  (Learns where aneurysms are located)                           │      │
│  │      │                                                           │      │
│  │      ▼                                                           │      │
│  │  Output: 14 Heatmaps (13 locations + "any aneurysm")            │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                     │
│       ▼                                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │ POST-PROCESSING                                                   │      │
│  │ • Apply Sigmoid (convert to 0-1 probabilities)                   │      │
│  │ • Find max probability in each heatmap                          │      │
│  │ • Extract coordinates of hottest voxel                          │      │
│  │ • Apply Left/Right anatomical correction                         │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│       │                                                                     │
│       ▼                                                                     │
│  RESULTS:                                                                  │
│  • "Aneurysm detected at Right MCA with 85% confidence"                   │
│  • Slice images showing exact location                                     │
│  • Risk assessment (High/Moderate/Low)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 0.16 Key Terms Glossary

| Term | Simple Explanation |
|------|-------------------|
| **Aneurysm** | Balloon-like bulge in blood vessel |
| **DICOM** | Medical image file format |
| **Voxel** | 3D pixel (cube instead of square) |
| **CNN** | Neural network that processes images |
| **U-Net** | U-shaped network for segmentation |
| **ResNet** | Network with skip connections |
| **nnU-Net** | Self-configuring medical imaging framework |
| **Blob Regression** | Predict heatmaps instead of boxes |
| **EDT** | Euclidean Distance Transform (smooth gradient) |
| **BCE Loss** | Binary Cross-Entropy (measures prediction error) |
| **TopK** | Focus on hardest 20% of predictions |
| **Inference** | Using trained model on new data |
| **CUDA** | GPU computing platform by NVIDIA |
| **Epoch** | One pass through entire training data |
| **Batch** | Small group of samples processed together |
| **Patch** | Small piece of large image |
| **Sliding Window** | Move patch across image to process all parts |

---

## 1. Problem Statement

### Competition Overview
**RSNA 2025 Intracranial Aneurysm Detection Challenge**
- **Host:** Radiological Society of North America (RSNA)
- **Platform:** Kaggle
- **Task:** Multi-label binary classification of intracranial aneurysms across 13 anatomical locations

### Clinical Objective
Develop an automated deep learning system to detect intracranial aneurysms from 3D medical imaging (CTA/MRA scans) and localize them to specific cerebral arteries.

### Why This Matters
- **Intracranial aneurysms** are pathological dilations (balloon-like bulges) in cerebral blood vessels
- **3-5%** of the general population has an unruptured brain aneurysm
- **Ruptured aneurysms** cause subarachnoid hemorrhage with **50% mortality rate**
- **Early detection** is critical for patient survival and treatment planning
- **Manual screening** is time-consuming and prone to human error

### Competition Metric
- **Weighted Log Loss** across 14 classes (13 locations + "Aneurysm Present")
- Lower is better

---

## 2. Medical Background

### What is an Intracranial Aneurysm?
An intracranial aneurysm is a localized, abnormal, balloon-like dilation of a blood vessel in the brain. They typically occur at arterial bifurcations in the Circle of Willis.

```
                    ┌─────────────────────────────────────┐
                    │       CIRCLE OF WILLIS              │
                    │   (Base of the Brain Vasculature)   │
                    └─────────────────────────────────────┘
                    
                              ACA (Anterior)
                                   │
                           ACoA ───┼─── ACoA
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                 L-MCA                          R-MCA
                    │                             │
                 L-ICA ─────────┬──────────── R-ICA
                    │           │               │
               L-PComA ─────────┼────────── R-PComA
                    │           │               │
                    └───── Basilar Tip ─────────┘
                              │
                    Other Posterior Circulation
```

### 13 Anatomical Locations Tracked

| # | Location | Abbreviation | Description |
|---|----------|--------------|-------------|
| 1 | Anterior Communicating Artery | ACoA | Most common site (~40% of aneurysms) |
| 2 | Left Middle Cerebral Artery | L-MCA | Lateral brain hemisphere supply |
| 3 | Right Middle Cerebral Artery | R-MCA | Lateral brain hemisphere supply |
| 4 | Left Supraclinoid ICA | L-Supra ICA | Above the clinoid process |
| 5 | Right Supraclinoid ICA | R-Supra ICA | Above the clinoid process |
| 6 | Left Infraclinoid ICA | L-Infra ICA | Below the clinoid process |
| 7 | Right Infraclinoid ICA | R-Infra ICA | Below the clinoid process |
| 8 | Left Posterior Communicating Artery | L-PComA | Connects anterior & posterior |
| 9 | Right Posterior Communicating Artery | R-PComA | Connects anterior & posterior |
| 10 | Left Anterior Cerebral Artery | L-ACA | Frontal lobe supply |
| 11 | Right Anterior Cerebral Artery | R-ACA | Frontal lobe supply |
| 12 | Basilar Tip | BA Tip | Top of basilar artery |
| 13 | Other Posterior Circulation | OPC | Vertebral, PICA, etc. |

### Imaging Modalities

| Modality | Full Name | Characteristics |
|----------|-----------|-----------------|
| **CTA** | CT Angiography | X-ray based, fast, uses contrast dye |
| **MRA** | MR Angiography | Magnetic resonance, no radiation |
| **MRI T1** | T1-weighted MRI | Shows anatomy, bright fat |
| **MRI T2** | T2-weighted MRI | Shows pathology, bright fluid |

---

## 3. Dataset Overview

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Series | 4,348 |
| Positive Cases (Aneurysm Present) | 1,864 (42.9%) |
| Negative Cases | 2,484 (57.1%) |
| Anatomical Locations | 13 |
| Dataset Size | ~215 GB |

### Modality Distribution

```
CTA:        ████████████████████ 1,808 (41.6%)
MRA:        ██████████████ 1,252 (28.8%)
MRI T2:     ███████████ 983 (22.6%)
MRI T1post: ███ 305 (7.0%)
```

### Age Distribution

```
<30:   █ 39
30-39: ██ 217
40-49: ██████ 614
50-59: ████████████ 1,165
60-69: █████████████ 1,321
70-79: ████████ 762
80+:   ██ 230

Mean Age: 58.5 years | Median: 59 years
```

### Location Prevalence (Among Positive Cases)

```
Anterior Communicating Artery:     ████████████████████ 363 (19.5%)
Left Supraclinoid ICA:             ██████████████████ 331 (17.8%)
Right MCA:                         ████████████████ 294 (15.8%)
Right Supraclinoid ICA:            ███████████████ 277 (14.9%)
Left MCA:                          ████████████ 219 (11.8%)
Other Posterior Circulation:       ██████ 113 (6.1%)
Basilar Tip:                       ██████ 110 (5.9%)
Right PComA:                       █████ 101 (5.4%)
Right Infraclinoid ICA:            █████ 98 (5.3%)
Left PComA:                        ████ 86 (4.6%)
Left Infraclinoid ICA:             ████ 78 (4.2%)
Right ACA:                         ███ 56 (3.0%)
Left ACA:                          ██ 46 (2.5%)
```

### Class Imbalance Challenge
- The dataset has significant class imbalance
- ACoA is most common (~19.5%), while Left ACA is rare (~2.5%)
- This affects model training and requires special handling

---

## 4. Solution Architecture

### 7th Place Solution (MIC-DKFZ Team)

We integrated the **7th place solution** from the Kaggle competition, developed by the **Division of Medical Image Computing at German Cancer Research Center (DKFZ)**.

#### Key Innovation: Blob Regression with nnU-Net

Instead of traditional object detection, the task is formulated as **multi-channel heatmap/blob regression**:

```
Traditional Detection:          Blob Regression:
┌─────────────────────┐        ┌─────────────────────┐
│  Find bounding box  │        │  Predict heatmap    │
│  around aneurysm    │   VS   │  where aneurysm is  │
│  + classify         │        │  hottest point      │
└─────────────────────┘        └─────────────────────┘
```

#### Why Blob Regression?

1. **No anchor boxes needed** - Simplifies architecture
2. **Works with nnU-Net** - Leverages proven medical image segmentation framework
3. **Natural for localization** - Heat represents probability
4. **Handles variable sizes** - Aneurysms vary from 2mm to 25mm+

### High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE SYSTEM ARCHITECTURE                         │
└──────────────────────────────────────────────────────────────────────────────┘

               FRONTEND (React)                    BACKEND (FastAPI)
            ┌─────────────────────┐            ┌─────────────────────┐
            │                     │            │                     │
            │  ┌───────────────┐  │            │  ┌───────────────┐  │
 User ──────│─▶│ Upload DICOM  │  │───HTTP────▶│  │ /analyze API  │  │
            │  └───────────────┘  │            │  └───────────────┘  │
            │         │           │            │         │           │
            │         ▼           │            │         ▼           │
            │  ┌───────────────┐  │            │  ┌───────────────┐  │
            │  │ WebSocket     │  │◀──Logs────│  │ Preprocessing │  │
            │  │ Log Terminal  │  │            │  │ Pipeline      │  │
            │  └───────────────┘  │            │  └───────────────┘  │
            │         │           │            │         │           │
            │         ▼           │            │         ▼           │
            │  ┌───────────────┐  │            │  ┌───────────────┐  │
 User ◀─────│──│ Results View  │  │◀──JSON────│  │ nnU-Net       │  │
            │  │ + Slice Images│  │            │  │ Inference     │  │
            │  └───────────────┘  │            │  └───────────────┘  │
            │                     │            │                     │
            └─────────────────────┘            └─────────────────────┘
                    Port 5174                         Port 8001
```

---

## 5. Preprocessing Pipeline

### Step 1: DICOM Loading

DICOM (Digital Imaging and Communications in Medicine) files are loaded and processed:

```python
# Parallel DICOM reading with PyDICOM
def process_series(input_folder):
    # 1. Read all DICOM files in parallel
    # 2. Sort by ImagePositionPatient or InstanceNumber
    # 3. Stack into 3D volume
    # 4. Extract spacing, origin, direction
    # 5. Reorient to RAS (Right-Anterior-Superior)
    return sitk_image
```

**RAS Orientation Standard:**
- **R** (Right): X-axis points to patient's right
- **A** (Anterior): Y-axis points forward
- **S** (Superior): Z-axis points upward (head)

### Step 2: Region of Interest (ROI) Cropping

To reduce computation, a **200 × 160 × 160 mm** ROI is extracted around the Circle of Willis:

```
Original Full Head+Neck Scan          Cropped ROI (Circle of Willis)
┌─────────────────────────┐          ┌─────────────────────────┐
│         Head            │          │                         │
│    ┌─────────────┐      │    ──▶   │   ○ Circle of Willis    │
│    │   Brain     │      │          │   ○ Aneurysm Location   │
│    │ ○ Aneurysm  │      │          │                         │
│    └─────────────┘      │          └─────────────────────────┘
│         Neck            │
│         Body            │           200mm (Z) × 160mm (Y) × 160mm (X)
└─────────────────────────┘
```

**Why crop?**
- Full scans can be 500+ slices
- Most aneurysms are in the Circle of Willis region
- Reduces GPU memory requirements
- Speeds up inference significantly

### Step 3: Intensity Normalization (Z-Score)

```python
# Z-score normalization
normalized = (image - global_mean) / global_std

# Global statistics from training set:
# Mean: computed from all training volumes
# Std: computed from all training volumes
```

### Step 4: Resampling to Target Spacing

All images are resampled to consistent voxel spacing:

```
Target Spacing: [0.70, 0.47, 0.47] mm (Z, Y, X)

Original:        512 × 512 × 300 slices @ variable spacing
Resampled:       ~223 × 343 × 342 voxels @ [0.70, 0.47, 0.47] mm
```

### Preprocessing Summary

```
┌────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [DICOM Files] ──▶ [Load & Sort] ──▶ [Stack to 3D Volume]         │
│                                              │                     │
│                                              ▼                     │
│                              [Orient to RAS Coordinate System]     │
│                                              │                     │
│                                              ▼                     │
│                              [Crop ROI: 200×160×160 mm]           │
│                                              │                     │
│                                              ▼                     │
│                              [Resample to Target Spacing]          │
│                              [0.70, 0.47, 0.47] mm                │
│                                              │                     │
│                                              ▼                     │
│                              [Z-Score Normalization]               │
│                                              │                     │
│                                              ▼                     │
│                              [Output: Preprocessed Tensor]         │
│                              Shape: (1, Z, Y, X)                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 6. Model Architecture

### nnU-Net Framework

**nnU-Net** (no-new-Net) is a self-configuring framework for medical image segmentation developed by DKFZ. It automatically:
- Determines optimal preprocessing
- Selects architecture parameters
- Configures training schedule

### Residual Encoder U-Net (ResEnc)

```
                         RESIDUAL ENCODER U-NET ARCHITECTURE
    
    Input                                                              Output
  (1,Z,Y,X)                                                          (14,Z,Y,X)
      │                                                                  ▲
      ▼                                                                  │
  ┌───────────┐                                                    ┌───────────┐
  │  Stage 1  │ ─────────────────────────────────────────────────▶ │  Stage 1  │
  │  32 feat  │                    Skip Connection                 │  Decoder  │
  └───────────┘                                                    └───────────┘
      │ ↓2x                                                            ▲
      ▼                                                                │
  ┌───────────┐                                                    ┌───────────┐
  │  Stage 2  │ ─────────────────────────────────────────────────▶ │  Stage 2  │
  │  64 feat  │                                                    │  Decoder  │
  └───────────┘                                                    └───────────┘
      │ ↓2x                                                            ▲
      ▼                                                                │
  ┌───────────┐                                                    ┌───────────┐
  │  Stage 3  │ ─────────────────────────────────────────────────▶ │  Stage 3  │
  │ 128 feat  │                                                    │  Decoder  │
  └───────────┘                                                    └───────────┘
      │ ↓2x                                                            ▲
      ▼                                                                │
  ┌───────────┐                                                    ┌───────────┐
  │  Stage 4  │ ─────────────────────────────────────────────────▶ │  Stage 4  │
  │ 256 feat  │                                                    │  Decoder  │
  └───────────┘                                                    └───────────┘
      │ ↓2x                                                            ▲
      ▼                                                                │
  ┌───────────┐                                                    ┌───────────┐
  │  Stage 5  │ ─────────────────────────────────────────────────▶ │  Stage 5  │
  │ 320 feat  │                                                    │  Decoder  │
  └───────────┘                                                    └───────────┘
      │ ↓2x                                                            ▲
      ▼                                                                │
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                           BOTTLENECK (Stage 6)                           │
  │                              320 features                                │
  └───────────────────────────────────────────────────────────────────────────┘
```

### Architecture Specifications

| Parameter | Value |
|-----------|-------|
| **Architecture** | Residual Encoder U-Net (3D) |
| **Stages** | 6 |
| **Features per Stage** | [32, 64, 128, 256, 320, 320] |
| **Kernel Size** | 3×3×3 (all stages) |
| **Patch Size** | 96 × 160 × 128 voxels |
| **Output Channels** | 14 (13 locations + background) |
| **Convolution Type** | 3D Conv (torch.nn.Conv3d) |
| **Normalization** | Instance Normalization |
| **Activation** | LeakyReLU |

### Blob Regression Target

Ground truth aneurysms are converted to **EDT Blobs** (Euclidean Distance Transform spheres):

```
Traditional Segmentation Mask:          EDT Blob Target:
┌──────────────────────────┐           ┌──────────────────────────┐
│                          │           │                          │
│        ████              │           │        ░░░░              │
│       ██████             │    ──▶    │       ░████░             │
│       ██████             │           │       ░████░             │
│        ████              │           │        ░░░░              │
│                          │           │                          │
└──────────────────────────┘           └──────────────────────────┘
   Binary mask (0/1)                    Smooth gradient (0.0-1.0)
```

**EDT Blob Properties:**
- Radius: 65 voxels (optimal from experiments)
- Values: 1.0 at center, decaying to 0.0 at edges
- Shape: 3D sphere with smooth Euclidean distance transform

---

## 7. Training Process

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 32 |
| **Patch Size** | 96 × 160 × 128 |
| **Epochs** | 3000 (1500 used in final submission) |
| **Iterations per Epoch** | 250 |
| **Initial Learning Rate** | 0.01 |
| **Optimizer** | SGD with momentum |
| **LR Schedule** | Polynomial decay |
| **Hardware** | 4× NVIDIA A100 40GB |
| **Training Time** | ~4.5 days |

### Loss Function: TopK BCE

The model uses **Binary Cross-Entropy (BCE)** computed only on the **top 20% worst-performing voxels**:

```python
# TopK BCE Loss
def topk_bce_loss(predictions, targets):
    # 1. Compute BCE for all voxels
    bce_all = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
    
    # 2. Find the top 20% hardest voxels (highest loss)
    k = int(0.2 * bce_all.numel())
    topk_losses, _ = torch.topk(bce_all.flatten(), k)
    
    # 3. Return mean of hardest voxels
    return topk_losses.mean()
```

**Why TopK?**
- Most voxels are easy (background)
- Focuses learning on difficult regions
- Prevents easy negatives from dominating

### Data Augmentation

| Augmentation | Description |
|--------------|-------------|
| **Spatial Transforms** | Rotation, scaling, elastic deformation |
| **Intensity Transforms** | Brightness, contrast, gamma |
| **Noise** | Gaussian noise, blur |
| **Note** | Left/Right mirroring disabled (class labels are lateralized) |

### Training Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [Preprocessed Data] ──▶ [Data Loader] ──▶ [Augmentation]         │
│                                                │                   │
│                                                ▼                   │
│                                   [Extract Random Patch]           │
│                                   96 × 160 × 128 voxels            │
│                                                │                   │
│                                                ▼                   │
│                                   [Forward Pass through Model]     │
│                                                │                   │
│                                                ▼                   │
│                                   [Compute TopK BCE Loss]          │
│                                   (on top 20% hardest voxels)      │
│                                                │                   │
│                                                ▼                   │
│                    [Backpropagation + SGD Update]                 │
│                                                │                   │
│                                                ▼                   │
│                    [PolyLR Schedule: lr = lr₀ × (1 - epoch/max)^0.9]│
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Cross-Validation Strategy

```
5-Fold Cross-Validation (Stratified by Modality)

Fold 1: ████████░░ Train | ██ Val (CTA/MRA balanced)
Fold 2: ██████░░██ Train | ██ Val (CTA/MRA balanced)
Fold 3: ████░░████ Train | ██ Val (CTA/MRA balanced)
Fold 4: ██░░██████ Train | ██ Val (CTA/MRA balanced)
Fold 5: ░░████████ Train | ██ Val (CTA/MRA balanced)

Final Model: Trained on ALL data ("fold_all")
```

---

## 8. Inference Pipeline

### Step-by-Step Inference

```
┌────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Step 1: LOAD DICOM                                               │
│  ─────────────────                                                │
│  • Read DICOM series from upload folder                           │
│  • Detect modality (CTA/MRA/MRI)                                  │
│  • Stack into 3D volume                                           │
│                          │                                         │
│                          ▼                                         │
│  Step 2: PREPROCESSING                                            │
│  ────────────────────                                             │
│  • Orient to RAS coordinate system                                │
│  • Crop to 200×160×160 mm ROI                                     │
│  • Resample to [0.70, 0.47, 0.47] mm spacing                      │
│  • Z-score normalization                                          │
│                          │                                         │
│                          ▼                                         │
│  Step 3: SLIDING WINDOW INFERENCE                                 │
│  ───────────────────────────────                                  │
│  • Split volume into overlapping patches (96×160×128)             │
│  • Process each patch through model                               │
│  • Step size: 50% overlap                                         │
│                          │                                         │
│                          ▼                                         │
│  Step 4: AGGREGATE PREDICTIONS                                    │
│  ─────────────────────────────                                    │
│  • Max-aggregate across all patches per channel                   │
│  • Apply sigmoid activation                                       │
│  • Output: 14-channel probability map                             │
│                          │                                         │
│                          ▼                                         │
│  Step 5: POST-PROCESSING                                          │
│  ──────────────────────                                           │
│  • Apply L/R anatomical correction                                │
│  • Extract max probability per location                           │
│  • Find coordinates of hottest voxels                             │
│  • Generate visualization images                                  │
│                          │                                         │
│                          ▼                                         │
│  Step 6: RETURN RESULTS                                           │
│  ─────────────────────                                            │
│  • 13 location probabilities                                      │
│  • Risk level (High/Moderate/Low)                                 │
│  • Slice images with bounding boxes                               │
│  • Coordinate information                                         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Probability Extraction

```python
# For each anatomical location channel
for channel in range(14):
    # Get 3D probability heatmap for this location
    heatmap = sigmoid(logits[channel])  # Shape: (Z, Y, X)
    
    # Take maximum as final probability
    probability = heatmap.max()
    
    # Find location of maximum
    z, y, x = np.unravel_index(heatmap.argmax(), heatmap.shape)
```

### Anatomical Left/Right Correction

A known issue in the dataset requires swapping Left/Right predictions:

```python
# Swap lateral pairs due to training data orientation
pairs = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
for left_idx, right_idx in pairs:
    probs[:, left_idx], probs[:, right_idx] = probs[:, right_idx], probs[:, left_idx]
```

---

## 9. Application Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FULL SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         FRONTEND (React + Vite)                      │  │
│  │                            Port: 5174                                │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐   │  │
│  │  │    Home     │ │  Analysis   │ │   Results   │ │ Architecture │   │  │
│  │  │   (Info)    │ │ (Upload UI) │ │  (Viewer)   │ │   (Docs)     │   │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └──────────────┘   │  │
│  │                         │            ▲                               │  │
│  │                         │ HTTP POST  │ JSON                         │  │
│  │                         │            │                               │  │
│  └─────────────────────────┼────────────┼───────────────────────────────┘  │
│                            │            │                                  │
│                            ▼            │                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      BACKEND (FastAPI + Python)                       │  │
│  │                            Port: 8001                                 │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐    │  │
│  │  │ /analyze    │ │  /health    │ │ /ws/logs    │ │  /dataset    │    │  │
│  │  │  (POST)     │ │   (GET)     │ │ (WebSocket) │ │   (GET)      │    │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └──────────────┘    │  │
│  │         │                                                             │  │
│  │         ▼                                                             │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │                    INFERENCE ENGINE                             │  │  │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐   │  │  │
│  │  │  │ Preprocessor│─▶│ nnUNet     │─▶│ Post-processing &      │   │  │  │
│  │  │  │ (DICOM→3D)  │ │ Predictor  │ │ Visualization           │   │  │  │
│  │  │  └─────────────┘ └─────────────┘ └─────────────────────────┘   │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                   │                                         │
│                                   ▼                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      MODEL CHECKPOINT (PyTorch)                        │  │
│  │           checkpoint_epoch_1500.pth (~400MB)                          │  │
│  │           + dataset.json + plans.json                                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Frontend Components

| Page | File | Purpose |
|------|------|---------|
| **Home** | `Home.jsx` | Project introduction, abstract, quick links |
| **Analysis** | `Analysis.jsx` | DICOM upload, real-time log terminal |
| **Results** | `Results.jsx` | Slice viewer, detection gallery, risk assessment |
| **Architecture** | `Architecture.jsx` | Technical documentation |
| **Dataset** | `Dataset.jsx` | Dataset statistics visualization |

### Backend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/analyze` | POST | Upload DICOM, run inference |
| `/analysis/{id}` | GET | Retrieve analysis results |
| `/locations` | GET | List anatomical locations |
| `/demo/predict` | POST | Demo mode (simulation) |
| `/ws/logs` | WebSocket | Real-time inference logs |
| `/dataset/info` | GET | Dataset statistics |

### Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React 18, Vite, TailwindCSS, Framer Motion, Axios |
| **Backend** | Python 3.10, FastAPI, Uvicorn, Pydantic |
| **ML Framework** | PyTorch 2.1, nnU-Net v2 |
| **Medical Imaging** | SimpleITK, PyDICOM, nibabel |
| **Visualization** | Matplotlib, scikit-image |

---

## 10. Results & Metrics

### Competition Performance

| Metric | Score |
|--------|-------|
| **Public Leaderboard** | 0.83 |
| **Private Leaderboard** | 0.83 |
| **Final Position** | 7th place |
| **Internal Validation** | ~0.90 |

### Per-Location Performance (Internal Validation)

```
Location                          Dice Score
────────────────────────────────────────────
Anterior Communicating Artery     ████████░ 0.89
Left Supraclinoid ICA             ████████░ 0.91
Right Supraclinoid ICA            ████████░ 0.88
Left MCA                          ████████░ 0.87
Right MCA                         ████████░ 0.90
Basilar Tip                       ███████░░ 0.85
Other Posterior Circulation       ██████░░░ 0.78
Left/Right PComA                  ███████░░ 0.82
Left/Right Infraclinoid ICA       ███████░░ 0.80
Left/Right ACA                    ██████░░░ 0.75
```

### Blob Size Optimization (Fold-0)

| EDT Radius | Validation Score |
|------------|------------------|
| 25 voxels | 0.888 |
| 35 voxels | 0.876 |
| 45 voxels | 0.893 |
| 55 voxels | 0.883 |
| **65 voxels** | **0.896** (Best) |
| 75 voxels | 0.871 |
| 85 voxels | 0.871 |
| 95 voxels | 0.866 |

---

## 11. Challenges Faced

### Challenge 1: Environment Conflicts

**Problem:** Python virtual environment (venv) conflicting with Conda environment
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:** Renamed venv folder and used Conda exclusively
```powershell
Rename-Item "venv" "venv_backup"
conda activate pretrained_detect
python -m uvicorn main:app --reload --port 8001
```

### Challenge 2: Missing Dependencies

**Problem:** scikit-image not found during import
```
ModuleNotFoundError: No module named 'skimage'
```

**Solution:** Install in correct Conda environment
```bash
conda activate pretrained_detect
pip install scikit-image pydicom
```

### Challenge 3: GPU Memory Limitations

**Problem:** Large 3D volumes caused CUDA out-of-memory errors

**Solution:**
1. Cropped ROI to 200×160×160 mm
2. Used `perform_everything_on_device=False` for CPU fallback
3. Cleared GPU cache between operations

### Challenge 4: Left/Right Label Swap

**Problem:** Model consistently predicted wrong anatomical side

**Solution:** Applied systematic L/R swap correction in post-processing
```python
# Swap all lateral pairs
for left_idx, right_idx in [(3,4), (5,6), (7,8), (9,10), (11,12)]:
    probs[:, left_idx], probs[:, right_idx] = probs[:, right_idx].clone(), probs[:, left_idx].clone()
```

### Challenge 5: Multi-Modality Handling

**Problem:** Different imaging modalities (CTA, MRA, MRI) have different characteristics

**Solution:** 
- Used modality-agnostic Z-score normalization
- Stratified cross-validation by modality
- Model learned modality-invariant features

### Challenge 6: Class Imbalance

**Problem:** Some locations (Left ACA: 2.5%) much rarer than others (ACoA: 19.5%)

**Solution:**
- TopK BCE loss focuses on hardest voxels
- Blob regression naturally handles scale
- Oversample foreground patches during training

### Challenge 7: Kaggle Platform Instabilities

**Problem:** Submissions timing out despite identical code

**Solution (MIC-DKFZ team):**
- Used checkpoint from epoch 1500 instead of 3000
- Disabled test-time augmentation
- Single model inference (no ensemble)

---

## 12. Key Learnings

### Technical Learnings

1. **nnU-Net is Powerful**: Self-configuring framework significantly reduces development time for medical imaging tasks

2. **Blob Regression > Detection**: For sparse objects like aneurysms, heatmap regression outperforms anchor-based detection

3. **ROI Cropping is Critical**: Focusing on anatomically relevant regions improves both speed and accuracy

4. **Post-processing Matters**: Simple fixes like L/R swap correction can significantly impact results

5. **Environment Management**: Medical imaging projects have complex dependencies; Conda is preferred over pip/venv

### Medical Domain Learnings

1. **Anatomical Understanding**: Knowing the Circle of Willis structure helps validate predictions

2. **Multi-Modality**: CTA/MRA/MRI each have different strengths for aneurysm visualization

3. **Clinical Relevance**: Location-specific detection is crucial for surgical planning

### Project Management Learnings

1. **Start with Proven Solutions**: Adapting competition-winning solutions accelerates development

2. **Documentation is Essential**: Complex pipelines need clear documentation for maintenance

3. **End-to-End Testing**: Integration testing across frontend/backend/model catches issues early

---

## 13. How to Run

### Prerequisites

```bash
# Required Software
- Python 3.10+
- Node.js 18+
- Conda (Anaconda/Miniconda)
- CUDA 11.8+ (for GPU inference)
```

### Step 1: Set Up Conda Environment

```bash
# Create and activate environment
conda create -n pretrained_detect python=3.10 -y
conda activate pretrained_detect

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install fastapi uvicorn nnunetv2 SimpleITK pydicom nibabel scikit-image matplotlib
```

### Step 2: Start Backend

```powershell
# Navigate to backend
cd "C:\Users\Rayan\Desktop\Main Project\Code\Pretrained detection\backend"

# Activate conda environment
conda activate pretrained_detect

# Start server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

Expected output:
```
Initializing inference engine...
Device: cuda
✅ nnU-Net model loaded successfully!
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8001
```

### Step 3: Start Frontend

```powershell
# Open new terminal
cd "C:\Users\Rayan\Desktop\Main Project\Code\Pretrained detection\frontend"

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

Expected output:
```
VITE v5.4.21 ready in 473 ms
➜ Local: http://localhost:5174/
```

### Step 4: Access Application

Open browser: **http://localhost:5174**

### Usage Flow

1. Go to **Analysis** page
2. Drag and drop DICOM files (.dcm)
3. Click **Start Analysis**
4. Watch real-time logs in terminal
5. View results with slice images and risk assessment

---

## 14. References

### Papers & Publications

1. **nnU-Net**: Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." *Nature Methods* (2021)

2. **Residual U-Net**: Zhang, Z., et al. "Road extraction by deep residual u-net." *IEEE Geoscience and Remote Sensing Letters* (2018)

3. **Blob Detection**: Kong, H., et al. "A generalized Laplacian of Gaussian filter for blob detection and its applications." *IEEE Transactions on Cybernetics* (2013)

### Competition Resources

- **Kaggle Competition**: [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)
- **7th Place Solution**: [MIC-DKFZ Writeup](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/7th-place-solution)
- **Model Checkpoint**: [Kaggle Dataset](https://www.kaggle.com/datasets/st3v3d/rsna-2025-7th-place-checkpoint)
- **Solution Code**: [GitHub Repository](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution)

### Framework Documentation

- **nnU-Net Documentation**: [GitHub](https://github.com/MIC-DKFZ/nnUNet)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **React Documentation**: [react.dev](https://react.dev)

---

## Appendix A: File Structure

```
Pretrained detection/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── inference.py            # nnU-Net inference engine
│   ├── dataset_router.py       # Dataset statistics API
│   ├── requirements.txt        # Python dependencies
│   └── models/
│       └── checkpoint/
│           └── Dataset004_iarsna_crop/
│               └── Kaggle2025RSNATrainer.../
│                   ├── dataset.json
│                   ├── plans.json
│                   └── fold_all/
│                       └── checkpoint_epoch_1500.pth
│
├── frontend/
│   ├── package.json            # Node dependencies
│   ├── vite.config.js          # Vite configuration
│   └── src/
│       ├── App.jsx             # Main React app
│       ├── api/client.js       # API client
│       └── pages/
│           ├── Home.jsx
│           ├── Analysis.jsx
│           ├── Results.jsx
│           ├── Architecture.jsx
│           └── Dataset.jsx
│
├── sol/                        # nnU-Net solution code
│   └── nnunetv2/
│       ├── training/
│       │   └── nnUNetTrainer/
│       │       └── project_specific/
│       │           └── kaggle2025_rsna/
│       │               └── Kaggle2025RSNATrainer.py
│       └── dataset_conversion/
│           └── kaggle_2025_rsna/
│               └── official_data_to_nnunet.py
│
└── docker-compose.yml          # Docker deployment
```

---

## Appendix B: API Response Example

```json
{
  "id": "abc123",
  "status": "completed",
  "predictions": [
    {
      "location": "Anterior Communicating Artery",
      "probability": 0.8234,
      "detected": true,
      "coordinates": {"x": 171, "y": 186, "z": 112},
      "slice_number": 112
    },
    {
      "location": "Left Middle Cerebral Artery",
      "probability": 0.1542,
      "detected": false,
      "coordinates": null,
      "slice_number": null
    }
  ],
  "overall_risk": "High",
  "confidence": 0.8234,
  "processing_time": 45.23,
  "model_loaded": true,
  "modality": "CTA",
  "slice_images": [
    {
      "location": "Anterior Communicating Artery",
      "slice_z": 112,
      "filename": "IM00112.dcm",
      "image_base64": "data:image/png;base64,...",
      "probability": 0.8234,
      "bbox": [145, 160, 52, 52]
    }
  ]
}
```

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Author:** Rayan  
**Project:** RSNA Intracranial Aneurysm Detection System
