# Intracranial Aneurysm Detection - Project Schedule (10 Phases)
**Timeline:** December 8, 2025 - April 30, 2026

## 📅 High-Level Timeline

| Phase | Dates | Focus Area | Status |
|-------|-------|------------|--------|
| **1. Init & Data** | Dec 8 - Dec 21 | Data Pipeline & Visualization | ✅ Completed |
| **2. Baseline** | Dec 22 - Jan 15 | 3D ResNet Backbone & Basic Training | 🚧 In Progress |
| **3. Multi-Task** | Jan 16 - Jan 29 | ROI Extraction & Multi-Task U-Net | 📅 Scheduled |
| **4. Advanced ML** | Jan 30 - Feb 12 | ROI Classifier & Focal/Weighted Losses | 📅 Scheduled |
| **5. Seg. Optimization** | Feb 13 - Feb 26 | nnU-Net Training & Integration | 📅 Scheduled |
| **6. Validation** | Feb 27 - Mar 12 | Cross-Validation & Competition Metrics | 📅 Scheduled |
| **7. Backend API** | Mar 13 - Mar 26 | FastAPI Server & Inference Endpoints | 📅 Scheduled |
| **8. Frontend** | Mar 27 - Apr 9 | React Dashboard & Dicom Viewer | 📅 Scheduled |
| **9. Integration** | Apr 10 - Apr 23 | Full Stack Connection & UI Polish | 📅 Scheduled |
| **10. Finalization** | Apr 24 - Apr 30 | Testing, Documentation & Submission | 📅 Scheduled |

---

## 🗓️ Detailed Phase Breakdown

### Phase 1: Initiation & Data Setup (Dec 8 - Dec 21)
*Goal: Establish environment and master the dataset.*
*   **Dec 8 - 14:** Repository setup, Conda environment, Literature review (RSNA solutions). Download Kaggle dataset.
*   **Dec 15 - 21:** Implement `preprocessing.py` (DICOM loading, windowing). Create 3D visualization tools (`visualize_preprocessing.py`).

### Phase 2: Baseline Modeling (Dec 22 - Jan 15)
*Goal: Get an end-to-end training loop running and stable.*
*   **Dec 22 - 28:** Implement 3D ResNet backbone (`models/backbone.py`). Setup basic training loop (`train.py`).
*   **Dec 29 - Jan 4:** Implement 3D data augmentation (flip, rotate).
*   **Jan 5 - 15:** Debug training instability. Fine-tune baseline hyperparameters. Ensure loss convergence.

### Phase 3: Multi-Task Learning Strategy (Jan 16 - Jan 29)
*Goal: Move beyond simple classification to location-aware detection.*
*   **Jan 16 - 22:** Implement vessel segmentation dataset. Develop `MultiTaskUNet`.
*   **Jan 23 - 29:** Begin `ROIClassifier` implementation (Location-Aware Transformer). Design Vessel-Masked Pooling logic.

### Phase 4: Advanced Model Training (Jan 30 - Feb 12)
*Goal: Integrate 1st-place solution techniques.*
*   **Jan 30 - Feb 5:** Complete Transformer implementation. Train ROI classifier on full dataset (FP16).
*   **Feb 6 - 12:** Implement advanced loss functions (Focal Loss, Asymmetric Loss) to handle class imbalance.

### Phase 5: Segmentation Optimization (Feb 13 - Feb 26)
*Goal: Improve vessel segmentation quality to aid classification.*
*   **Feb 13 - 19:** Configure and train full nnU-Net on segmentation masks.
*   **Feb 20 - 26:** Integrate nnU-Net predictions as input channels/masks for the main classifier.

### Phase 6: Validation & Refinement (Feb 27 - Mar 12)
*Goal: Robust evaluation and metric boosting.*
*   **Feb 27 - Mar 5:** Implement 5-Fold Cross-Validation. Train models on all folds.
*   **Mar 6 - 12:** Implement competition metrics (Weighted AUC). Analyze failure cases (False Positives).

### Phase 7: Backend API Development (Mar 13 - Mar 26)
*Goal: Expose the model as a service.*
*   **Mar 13 - 19:** Setup FastAPI server structure. Create endpoints for file upload.
*   **Mar 20 - 26:** Wrap inference pipeline into a callable API service. Handle asynchronous inference tasks.

### Phase 8: Frontend Dashboard Creation (Mar 27 - Apr 9)
*Goal: Build the user interface.*
*   **Mar 27 - Apr 2:** Initialize React/Vite project with Tailwind CSS. Design main dashboard layout.
*   **Apr 3 - 9:** Implement medical image viewer component to display scans in the browser.

### Phase 9: Integration & Visualization (Apr 10 - Apr 23)
*Goal: Connect the pieces.*
*   **Apr 10 - 16:** Connect Frontend to Backend API. Display inference results on the viewer.
*   **Apr 17 - 23:** UI/UX Polish. Add loading states, error handling, and ensure responsive design.

### Phase 10: Finalization & Documentation (Apr 24 - Apr 30)
*Goal: Wrap up.*
*   **Apr 24 - 30:**
    *   System testing (end-to-end flows).
    *   Write final code documentation.
    *   Draft final paper/report sections.
    *   Prepare final submission.
