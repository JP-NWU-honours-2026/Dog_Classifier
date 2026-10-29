# ITRI626 Project: Clean vs. Unclean Dataset Experiment & Comparative Study

**Author**: JP  
**Model Architecture**: EfficientNetV2-S (Transfer Learning)  
**Dataset**: Stanford Dogs Dataset (120 Breeds)  
**Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA + AMP)  
**Status**: Ready for Execution  

---

## 1. Executive Summary & Required Methodology

The experiment follows the exact academic workflow established by Lindani to ensure scientific comparability:

1. **Run-Based Artifact Chain (`results/runs/<run_name>/`)**:
   - `train.py` executes and automatically logs:
     - `run_config.json`: Trainable parameters, hyperparameters, loss/optimizer settings, GPU model, CUDA version, start time.
     - `metrics.csv`: Real-time epoch-by-epoch trace (`epoch, lr, train_loss, train_acc, val_loss, val_acc, val_top5, val_macro_f1, seconds, best_so_far`).
     - `best_model.pth`: Checkpoint with highest validation accuracy.
     - `training_curves.png`: Visual loss and accuracy curves annotating phases.
   - `evaluate.py --run <run_name>` loads `best_model.pth` and evaluates the held-out test set once, outputting into `results/runs/<run_name>/eval_test/`:
     - `metrics.json` & `metrics_summary.txt`: Test Top-1, Top-5, Macro F1, Precision, Recall.
     - `confusion_matrix.png` & `confusion_matrix.csv`.
     - `per_class_metrics.csv` & `sample_predictions.png`.

2. **The 4 Core Comparison Metrics**:
   1. **Test Accuracy**: Top-1 Accuracy (%) and Top-5 Accuracy (%).
   2. **Macro F1-Score**: Unweighted average F1 across all 120 breeds.
   3. **Trainable Parameter Count**: Exact count of weights updated during training.
   4. **Training Time & Hardware**: Wall-clock training duration (minutes) and GPU utilized.

3. **Student Baseline Anchor (From-Scratch CNN)**:
   - **Architecture**: 5-block Convolutional Neural Network trained from scratch.
   - **Trainable Parameters**: ~4,000,000 (~4M).
   - **Top-1 Test Accuracy**: **53.88%**.
   - **Top-5 Test Accuracy**: **84.78%**.
   - **Macro F1-Score**: **0.5320 (53.20%)**.
   - **Hardware & Duration**: **64 minutes** on **Tesla T4 GPU** (60 epochs).

---

## 2. Directory Architecture for Results

```
JP_Model/
├── duplicate_check.py              # Adapted duplicate & leakage detection engine
├── create_clean_manifest.py        # Generates clean split from approved removals
├── dataset.py                      # Dataset loader supporting clean & unclean manifests
├── model.py                        # EfficientNetV2-S architecture definition
├── train.py                        # Two-phase transfer learning trainer with run logging
├── evaluate.py                     # Standalone test set evaluator targeting runs
├── split_manifest.json             # Unclean baseline split (20,580 images)
├── split_manifest_clean.json       # Clean curated split (deduplicated)
└── results/
    ├── duplicate_check/            # Duplicate analysis reports, CSVs & sample sheets
    │   ├── duplicate_check_summary.txt
    │   ├── candidate_duplicates.json # Full candidate pairs for visual approval
    │   ├── approved_removals.json    # User-approved removal list
    │   ├── leaked_photos.csv
    │   ├── confirmed_copies.csv
    │   └── label_conflicts.csv
    ├── runs/
    │   ├── unclean_transfer/       # Run 1: EfficientNetV2-S on split_manifest.json
    │   │   ├── run_config.json     # Hyperparameters, GPU, parameter count, metadata
    │   │   ├── metrics.csv         # Epoch-by-epoch training/val loss, acc, macro F1
    │   │   ├── best_model.pth      # Best checkpoint
    │   │   ├── training_curves.png # Dual loss & accuracy plots
    │   │   └── eval_test/          # Evaluation on held-out test split
    │   │       ├── metrics.json    # Top-1, Top-5, Macro F1, latency
    │   │       ├── metrics_summary.txt
    │   │       ├── confusion_matrix.png
    │   │       ├── confusion_matrix.csv
    │   │       ├── per_class_metrics.csv
    │   │       └── sample_predictions.png
    │   └── clean_transfer/         # Run 2: EfficientNetV2-S on split_manifest_clean.json
    │       ├── run_config.json
    │       ├── metrics.csv
    │       ├── best_model.pth
    │       ├── training_curves.png
    │       └── eval_test/
    │           ├── metrics.json
    │           ├── metrics_summary.txt
    │           ├── confusion_matrix.png
    │           ├── confusion_matrix.csv
    │           ├── per_class_metrics.csv
    │           └── sample_predictions.png
    └── comparison/                 # Benchmark Deliverables for Report Section 6/7
        ├── comparative_benchmark.csv# The 4 metrics across all models
        ├── comparative_curves.png  # Overlay curves (Scratch vs Unclean vs Clean)
        └── report_summary.md       # Formatted markdown table ready for project report
```

---

## 3. Comparative Benchmark Table

| Model | Architecture | Pretrained? | Trainable Params | Top-1 Test Acc | Top-5 Test Acc | Macro F1 | Training Time | Hardware |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **JP Baseline** | From-Scratch CNN | No | **~4.0M** | **53.88%** | **84.78%** | **0.5320** | 64 min | Tesla T4 GPU |
| **Lindani Baseline**| From-Scratch CNN | No | ~4.1M | ~50.5% | ~81.2% | ~0.4980 | ~70 min | Tesla T4 GPU |
| **JP Run 1 (Unclean)**| EfficientNetV2-S | Yes (ImageNet) | **20.3M** | **83.12%** | **97.75%** | **0.8254** | **14.74 min** | RTX 4050 GPU |
| **JP Run 2 (Clean)** | EfficientNetV2-S | Yes (ImageNet) | **20.3M** | **83.25%** | **98.07%** | **0.8268** | **14.20 min** | RTX 4050 GPU |

---

## 4. Step-by-Step Implementation Roadmap

```
├── [x] Phase 1: Duplicate & Conflict Detection Engine
│   ├── [x] 1.1 Adapt duplicate_check.py into JP_Model/
│   ├── [x] 1.2 Run duplicate_check.py (173 confirmed pairs, 50 conflicts, 74 cross-split leaks)
│   └── [x] 1.3 Export candidate_duplicates.json for visual review
│
├── [x] Phase 2: Interactive Visual Review & Approval Interface
│   ├── [x] 2.1 Add visual Duplicate & Conflict Review tab in Dog_Classifier_App (http://localhost:8000)
│   ├── [x] 2.2 Provide side-by-side image comparison with 1-click Approve/Reject
│   ├── [x] 2.3 Allow batch/individual approval across all splits
│   └── [x] 2.4 Save approved removals to JP_Model/results/duplicate_check/approved_removals.json
│
├── [x] Phase 3: Clean Manifest Generation
│   ├── [x] 3.1 Reviewed candidate duplicate/conflict pairs in browser Data Cleaning Studio
│   ├── [x] 3.2 Generated split_manifest_clean.json (20,410 total photos; 170 duplicates removed)
│   └── [x] 3.3 Verified split_manifest_clean.json integrity and 120-breed stratification (120/120 in train, val, test)
│
├── [x] Phase 4: Training Pipeline Enhancement (Run Logging & Real-Time Traces)
│   ├── [x] 4.1 Update train.py to log results/runs/<name>/run_config.json
│   ├── [x] 4.2 Stream real-time epoch metrics to results/runs/<name>/metrics.csv
│   ├── [x] 4.3 Automatically generate dual loss and accuracy curves upon completion
│   ├── [x] 4.4 Update evaluate.py to take --run <name> and output to eval_test/
│   └── [x] 4.5 Create compare_runs.py integrating the 4 comparison dimensions and student's scratch baseline
│
├── [x] Phase 5: Train Model A (Unclean Run: unclean_transfer) from Scratch
│   ├── [x] 5.1 Run 2-phase training on split_manifest.json with live metrics
│   ├── [x] 5.2 Save checkpoint and curves into results/runs/unclean_transfer/
│   └── [x] 5.3 Run evaluate.py --run unclean_transfer on test set (83.12% Top-1, 97.75% Top-5, 0.8254 F1)
│
├── [x] Phase 6: Train Model B (Clean Run: clean_transfer) from Scratch
│   ├── [x] 6.1 Run 2-phase training on split_manifest_clean.json with live metrics (14.20 min)
│   ├── [x] 6.2 Save checkpoint and curves into results/runs/clean_transfer/
│   └── [x] 6.3 Run evaluate.py --run clean_transfer on test set (83.25% Top-1, 98.07% Top-5, 0.8268 F1)
│
└── [x] Phase 7: Benchmark Synthesis & Report Generation
    ├── [x] 7.1 Cross-evaluate Model A on clean test set & Model B on clean test set
    ├── [x] 7.2 Compile comparative_benchmark.csv comparing all 4 models
    └── [x] 7.3 Generate overlay learning curves comparing Scratch vs Unclean vs Clean (comparative_curves.png)
```

---

## 5. Command Reference

```bash
# 1. Run Duplicate Check
python JP_Model/duplicate_check.py

# 2. Build Clean Split Manifest after visual approval
python JP_Model/create_clean_manifest.py

# 3. Train Model A (Unclean) from scratch with run logging
python JP_Model/train.py --run-name unclean_transfer --manifest JP_Model/split_manifest.json

# 4. Evaluate Model A on Test Split
python JP_Model/evaluate.py --run unclean_transfer

# 5. Train Model B (Clean) from scratch with run logging
python JP_Model/train.py --run-name clean_transfer --manifest JP_Model/split_manifest_clean.json

# 6. Evaluate Model B on Test Split
python JP_Model/evaluate.py --run clean_transfer

# 7. Generate Full Comparative Benchmark
python JP_Model/compare_runs.py
```

