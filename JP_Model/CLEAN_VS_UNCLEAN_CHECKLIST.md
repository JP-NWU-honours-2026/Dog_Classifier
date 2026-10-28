# ITRI626 Project: Clean vs. Unclean Dataset Experiment & Comparative Study

**Author**: JP  
**Model Architecture**: EfficientNetV2-S (Transfer Learning)  
**Dataset**: Stanford Dogs Dataset (120 Breeds)  
**Target Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA + AMP)  
**Status**: Ready for Execution  

---

## 1. Executive Summary & User Requirements

The user selected:
1. **Interactive Visual Approval**: The user must be able to visually inspect candidate duplicate and mislabeled photographs side-by-side and approve or reject them one-by-one.
2. **Global Deduplication**: Remove confirmed duplicates and mislabeled conflicts across **all splits** (train, validation, and test), not just cross-split leakage.
3. **Dual Retraining from Scratch**: Retrain both Model A (Unclean) and Model B (Clean) from scratch with real-time learning traces (epoch loss, training accuracy, validation accuracy, Top-5, learning rate).
4. **Structured Results Organization**: Neatly organize all outputs in `JP_Model/results/` under `duplicate_check/`, `unclean/`, `clean/`, and `comparison/`.

---

## 2. Directory Architecture for Results

```
JP_Model/
├── duplicate_check.py              # Adapted duplicate & leakage detection engine
├── dataset.py                      # Dataset loader supporting clean & unclean manifests
├── model.py                        # EfficientNetV2-S architecture definition
├── train.py                        # Two-phase transfer learning trainer with tracking
├── evaluate.py                     # Evaluator computing Top-1, Top-5, Macro F1, Conf Matrix
├── split_manifest.json             # Unclean split (20,580 images)
├── split_manifest_clean.json       # Clean split (filtered of user-approved duplicates & conflicts)
└── results/
    ├── duplicate_check/            # Duplicate analysis reports, CSVs & sample sheets
    │   ├── duplicate_check_summary.txt
    │   ├── candidate_duplicates.json # Full candidate pairs for visual approval
    │   ├── approved_removals.json    # User-approved removal list
    │   ├── leaked_photos.csv
    │   ├── confirmed_copies.csv
    │   └── label_conflicts.csv
    ├── unclean/                    # Model A artifacts (trained on unclean split)
    │   ├── best_model.pth          # Weights checkpoint
    │   ├── history.json            # Epoch-by-epoch loss & accuracy traces
    │   ├── training_curves.png     # Loss and accuracy vs epoch curves
    │   ├── confusion_matrix.png    # 120x120 test confusion matrix
    │   ├── sample_predictions.png  # Qualitative prediction samples
    │   └── evaluation_metrics.json # Test metrics (Top-1, Top-5, Macro F1)
    ├── clean/                      # Model B artifacts (trained on clean split)
    │   ├── best_model.pth          # Weights checkpoint
    │   ├── history.json            # Epoch-by-epoch loss & accuracy traces
    │   ├── training_curves.png     # Loss and accuracy vs epoch curves
    │   ├── confusion_matrix.png    # 120x120 test confusion matrix
    │   ├── sample_predictions.png  # Qualitative prediction samples
    │   └── evaluation_metrics.json # Test metrics (Top-1, Top-5, Macro F1)
    └── comparison/                 # Comparative deliverables for Project Report
        ├── comparative_metrics.json# Side-by-side metrics table
        ├── comparative_curves.png  # Overlay curves (Unclean vs Clean learning)
        └── report_summary.md       # Formatted markdown table for report Section 6/7
```

---

## 3. Step-by-Step Implementation Roadmap

```
├── [x] Phase 1: Duplicate & Conflict Detection Engine
│   ├── [x] 1.1 Copy & adapt duplicate_check.py into JP_Model/
│   ├── [x] 1.2 Run duplicate_check.py to generate JP_Model/results/duplicate_check/
│   └── [x] 1.3 Export candidate duplicate pairs and label conflicts to JSON (173 confirmed pairs)
│
├── [x] Phase 2: Interactive Visual Review & Approval Interface
│   ├── [x] 2.1 Add visual Duplicate & Conflict Review tab in Dog_Classifier_App
│   ├── [x] 2.2 Provide side-by-side image comparison with 1-click Approve/Reject
│   ├── [x] 2.3 Allow batch/individual approval across all splits
│   └── [x] 2.4 Save approved removals to JP_Model/results/duplicate_check/approved_removals.json
│
├── [ ] Phase 3: Clean Manifest Generation
│   ├── [ ] 3.1 Script create_clean_manifest.py to filter approved removals across all splits
│   ├── [ ] 3.2 Verify split_manifest_clean.json integrity and stratification
│   └── [ ] 3.3 Log summary of removed photos per split and per breed
│
├── [ ] Phase 4: Training Pipeline Enhancement (Real-Time Learning Traces)
│   ├── [ ] 4.1 Enhance train.py to support --manifest and --output_dir
│   ├── [ ] 4.2 Implement epoch-by-epoch and batch progress tracing (Loss, Acc, Top-5, LR)
│   └── [ ] 4.3 Automatically generate dual loss and accuracy curves at end of training
│
├── [ ] Phase 5: Train Model A (Unclean) from Scratch
│   ├── [ ] 5.1 Run 2-phase training on split_manifest.json (Warmup + Fine-Tuning)
│   ├── [ ] 5.2 Log real-time learning curve traces
│   └── [ ] 5.3 Save checkpoint and history to JP_Model/results/unclean/
│
├── [ ] Phase 6: Train Model B (Clean) from Scratch
│   ├── [ ] 6.1 Run 2-phase training on split_manifest_clean.json (Warmup + Fine-Tuning)
│   ├── [ ] 6.2 Log real-time learning curve traces
│   └── [ ] 6.3 Save checkpoint and history to JP_Model/results/clean/
│
├── [ ] Phase 7: Dual Evaluation & Cross-Evaluation Benchmark
│   ├── [ ] 7.1 Evaluate Model A on Unclean & Clean test splits
│   ├── [ ] 7.2 Evaluate Model B on Clean & Unclean test splits
│   ├── [ ] 7.3 Generate confusion matrices and sample predictions in respective folders
│   └── [ ] 7.4 Compile comparative metrics into JP_Model/results/comparison/
```

---

## 4. Evaluation Matrix

| Experiment | Training Split | Evaluation Test Split | Research Question Answered |
| :--- | :--- | :--- | :--- |
| **Model A (Standard)** | Unclean (20,580) | Unclean Test (3,139) | Baseline performance with natural dataset leakage. |
| **Model A (Unbiased)** | Unclean (20,580) | Clean Test (User-Cleaned) | Does leakage in train artificially inflate test performance? |
| **Model B (Clean)** | Clean (User-Cleaned) | Clean Test (User-Cleaned) | True performance when trained and tested on noise-free data. |
| **Model B (Robustness)** | Clean (User-Cleaned) | Unclean Test (3,139) | How well does a clean model generalize to real-world noisy data? |

---

## 5. Command Reference

```bash
# 1. Run Duplicate Check
python JP_Model/duplicate_check.py

# 2. Build Clean Split Manifest after visual approval
python JP_Model/create_clean_manifest.py

# 3. Train Model A (Unclean) with real-time tracing
python JP_Model/train.py --manifest JP_Model/split_manifest.json --output_dir JP_Model/results/unclean

# 4. Train Model B (Clean) with real-time tracing
python JP_Model/train.py --manifest JP_Model/split_manifest_clean.json --output_dir JP_Model/results/clean

# 5. Dual & Cross Evaluations
python JP_Model/evaluate.py --checkpoint JP_Model/results/unclean/best_model.pth --manifest JP_Model/split_manifest.json --output_dir JP_Model/results/unclean
python JP_Model/evaluate.py --checkpoint JP_Model/results/clean/best_model.pth --manifest JP_Model/split_manifest_clean.json --output_dir JP_Model/results/clean

# 6. Generate Comparative Benchmark
python JP_Model/compare_runs.py
```
