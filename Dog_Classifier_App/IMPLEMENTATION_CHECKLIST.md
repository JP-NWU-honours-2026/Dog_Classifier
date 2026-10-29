# ITRI626 Mini-Project: JP_Model & Dog Classifier App Implementation Plan

**Student**: JP  
**Architecture**: Pretrained EfficientNetV2-S (Transfer Learning)  
**Dataset**: Stanford Dogs Dataset (120 Breeds, 20,580 Images)  
**Hardware Target**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA + AMP)  
**Status**: Ready to Execute

---

## Executive Summary

This roadmap outlines the complete pipeline to:

1. **Verify dataset parity** with Lindani's model to preserve scientific experimental integrity.
2. **Execute full two-phase transfer learning** with EfficientNetV2-S across all 120 breeds.
3. **Evaluate and benchmark** on the held-out test split (Top-1, Top-5, and Macro F1).
4. **Upgrade the CanineVision Web Application** to accept any uploaded dog image, calculate softmax probabilities across all 120 breeds, and display the **Top 5 predicted dog breeds with exact percentage breakdowns and animated progress bars**, along with 5 quick-test sample dog buttons.

---

## Implementation Roadmap

```
├── [x] Stage 1: Dataset Verification & Experimental Parity
├── [x] Stage 2: Full 120-Breed Transfer Learning Training (JP_Model - COMPLETED 85.09% Val Acc)
├── [x] Stage 3: Test Set Evaluation & Benchmark Artifact Generation (COMPLETED 83.12% Top-1, 97.75% Top-5)
├── [x] Stage 4: Web Application Backend Upgrade (Dog_Classifier_App)
├── [x] Stage 5: Web Application UI/UX Redesign, Top-5 Display & Breed Image Explorer
└── [x] Stage 6: End-to-End Validation & Live Testing (PASSED)
```

---

### Stage 1: Dataset Verification & Experimental Parity

Ensure complete scientific consistency between baseline and transfer learning models.

- [x] **Split Manifest Verification**: Confirm that [`JP_Model/split_manifest.json`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/split_manifest.json) is 100% bit-for-bit identical to [`Lindani_Model/split_manifest.json`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/Lindani_Model/split_manifest.json).
  - _Result_: MD5 checksum `07f51ae3b9b67263e83f136b9ba74757` verified identical across all 20,580 photographs.
  - _Distribution_: 14,355 Train (70%) / 3,086 Validation (15%) / 3,139 Test (15%).
- [x] **Data Pipeline Integrity**:
  - [x] Standardize resolution to $224 \times 224 \times 3$.
  - [x] Apply standard ImageNet channel normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
  - [x] Enforce domain-appropriate training augmentations (Horizontal Flip, $\pm 15^\circ$ Rotation, Color Jitter; no vertical flips).
  - [x] Keep validation and test pipelines completely clean of augmentations.
- [x] **PyTorch DataLoader Optimization**:
  - [x] Enable `pin_memory=True` and `non_blocking=True` for fast GPU transfer.
  - [x] Configure batch size of 32 for optimal memory utilization on RTX 4050 (6GB VRAM).

---

### Stage 2: Full 120-Breed Transfer Learning Training (`JP_Model`)

Train the 20.3M parameter **EfficientNetV2-S** model using a two-phase transfer learning schedule.

- [x] **Weight Initialization**:
  - [x] Reset base model to pristine `ImageNet-1K` pretrained weights (`EfficientNet_V2_S_Weights.DEFAULT`).
  - [x] Initialize custom 120-class classification head (`Linear(1280, 120)` with `Dropout(0.3)`).
- [x] **Phase 1: Classifier Head Warmup (Frozen Backbone)**:
  - [x] Freeze all convolutional stages (`requires_grad = False` on `features`).
  - [x] Train classifier head for **2 epochs** (`lr=3e-4`, AdamW, label smoothing 0.1).
  - [x] _Checkpoint_: Epoch 2 reached **78.28% validation accuracy** across all 120 breeds!
- [x] **Phase 2: Fine-Tuning (Unfrozen Upper Stages)**:
  - [x] Unfreeze upper feature stages 6 and 7 (`requires_grad = True`), keeping lower stages 0–5 frozen to preserve general edge/texture filters.
  - [x] Train for **3 epochs** with `lr=1e-5` and `CosineAnnealingLR`.
  - [x] Mixed Precision: Utilized PyTorch AMP (`torch.amp.autocast`) for fast GPU throughput (18.3 min total training time on RTX 4050).
  - [x] Peak Validation Accuracy achieved: **85.09%** across all 120 breeds.
- [x] **Artifact Checkpointing**:
  - [x] Save best model checkpoint based on peak validation accuracy to [`JP_Model/best_model.pth`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/best_model.pth).
  - [x] Record epoch metrics in [`JP_Model/history.json`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/history.json).
  - [x] Generate dual loss and accuracy curves in [`JP_Model/training_curves.png`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/training_curves.png) annotating the phase boundary.

---

### Stage 3: Test Set Evaluation & Benchmark Artifact Generation

Evaluate the trained checkpoint strictly once on the held-out test split.

- [x] **Execute Test Evaluation**: Run [`JP_Model/evaluate.py`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/evaluate.py) on the 3,139 held-out test images.
- [x] **Compute Core Metrics**:
  - [x] Top-1 Test Accuracy: **83.12%**.
  - [x] Top-5 Test Accuracy: **97.75%**.
  - [x] Macro Precision: **83.26%**, Macro Recall: **82.87%**, Macro F1-Score: **82.54%** across all 120 breeds.
- [x] **Generate Visual Deliverables**:
  - [x] [`JP_Model/confusion_matrix.png`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/confusion_matrix.png): Full $120 \times 120$ heatmap.
  - [x] [`JP_Model/sample_predictions.png`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/sample_predictions.png): Qualitative prediction grid with ground truth comparison.
  - [x] [`JP_Model/evaluation_metrics.json`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/evaluation_metrics.json): Serialized evaluation summary for report tables.

---

### Stage 4: Web Application Backend Upgrade (`Dog_Classifier_App`)

Refactor the FastAPI backend to serve the full 120-breed transfer learning model.

- [x] **Model Loading & State Compatibility ([`server.py`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/Dog_Classifier_App/server.py))**:
  - [x] Point model path to [`JP_Model/best_model.pth`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/JP_Model/best_model.pth).
  - [x] Support both list and dictionary representations of `class_names` (120 breeds).
  - [x] Ensure seamless state dict key mapping (`base_model.` prefix compatibility).
- [x] **API Endpoint Refactor (`POST /api/predict`)**:
  - [x] Process incoming image (JPEG/PNG/WEBP) via standard inference transform.
  - [x] Compute softmax probabilities across all 120 classes.
  - [x] Extract and return the **Top 5 predicted classes** with breed display names, percentages rounded to 1 decimal place, and inference latency in milliseconds.
- [x] **Metadata & Breed Endpoints**:
  - [x] `GET /api/info`: Return active architecture (`EfficientNetV2-S`), number of classes (120), device (`CUDA`/`CPU`), and checkpoint source.
  - [x] `GET /api/breeds`: Return all 120 breeds strictly sorted alphabetically (A-Z) by display name.
  - [x] `GET /api/breeds/{breed_id}/images`: Return all dataset images for any selected breed.
- [x] **Sample Images Caching**:
  - [x] Populate [`Dog_Classifier_App/static/samples/`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/Dog_Classifier_App/static/samples/) with 5 diverse real dog breed test images:
    1. Golden Retriever
    2. German Shepherd
    3. Siberian Husky
    4. French Bulldog
    5. Chihuahua

---

### Stage 5: Web Application UI/UX Redesign & Breed Image Explorer

Deliver a high-aesthetic, responsive, glassmorphic interface that highlights the Top 5 predictions and provides an interactive dataset explorer.

- [x] **Layout & Theme ([`index.html`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/Dog_Classifier_App/static/index.html), [`style.css`](file:///c:/Users/jpret/Documents/NWU/2026/Sem%202/ITRI626%20AI/Mini-Project/Dog_Classifier/Dog_Classifier_App/static/style.css))**:
  - [x] Modern dark glassmorphism aesthetic with subtle gradients and clean typography (Plus Jakarta Sans / JetBrains Mono).
  - [x] Mode tabs: **AI Classifier** and **Breed Image Explorer**.
- [x] **Classifier View**:
  - [x] Drag-and-drop zone and file picker accepting any image.
  - [x] 5 interactive sample dog buttons with emojis for instantaneous 1-click classification testing.
  - [x] Hero Result Card: Highlights the #1 predicted breed with large typography, confidence percentage, and latency badge.
  - [x] Top 5 Ranked Breakdown with animated progress bars, badges `#1` through `#5`, and percentage values.
- [x] **Breed Image Explorer View**:
  - [x] **Alphabetical Dropdown List**: Strictly sorted A-Z; photo count removed from option text (displays clean breed name only).
  - [x] **Page Photo Count**: Cleanly displays image count badge on the page banner (e.g. `150 Images`).
  - [x] **Smart Search Algorithm (`matchPhraseInText`)**:
    - [x] Checks if the position of a phrase is not present in the text (`indexOf === -1` excludes non-matches).
    - [x] First checks if the first few letters match (`startsWith` -> Priority 1).
    - [x] Checks word boundary prefix matches (Priority 2).
    - [x] Orders substring matches by position in text (Priority 3).
    - [x] Returns an ordered result with interactive suggestions dropdown and matched text highlighting.
  - [x] **Interactive Gallery Grid**: Lazy-loaded thumbnails with 1-click "Classify Photo" and full-resolution lightbox viewer.

---

### Stage 6: End-to-End Validation & Live Testing

- [x] Start the FastAPI server: `python Dog_Classifier_App/server.py`.
- [x] Verify `http://localhost:8000` loads properly.
- [x] Test uploading arbitrary external dog images.
- [x] Test each of the 5 sample buttons and verify that the Top 5 probabilities sum and render accurately.
- [x] Verify Breed Explorer dropdown is alphabetically sorted without photo count text.
- [x] Verify smart search ranking behavior (typing 'gol' ranks Golden Retriever #1, 'husky' ranks Siberian Husky).
- [x] Confirm no JavaScript console errors or backend exceptions.

---

## Command Reference

| Action                             | Command                                                                         |
| :--------------------------------- | :------------------------------------------------------------------------------ |
| **Verify Dataset Split**           | `python JP_Model/dataset.py`                                                    |
| **Run Quick Smoke Test (5% data)** | `python JP_Model/train.py --quick_test`                                         |
| **Run Full 120-Breed Training**    | `python JP_Model/train.py --head_epochs 2 --fine_tune_epochs 3 --batch_size 32` |
| **Evaluate Test Set**              | `python JP_Model/evaluate.py --checkpoint JP_Model/best_model.pth`              |
| **Run Web Application**            | `python Dog_Classifier_App/server.py`                                           |
