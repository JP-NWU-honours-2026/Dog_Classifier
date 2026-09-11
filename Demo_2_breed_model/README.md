# ITRI626 Deep Learning - 2-Breed Dog Classifier Demo

This folder contains a fully functional, reproducible binary classification demonstration pipeline distinguishing between **Chihuahuas** and **Siberian Huskies** from the Stanford Dogs dataset.

It serves as the reference template for the **ITRI626 practical project**, fulfilling all core rubric requirements:
- **Reproducibility**: Strict train/val/test splits (70/15/15) saved via `split_manifest.json` with fixed random seed (`seed=42`).
- **Data Leakage Prevention**: Split generated deterministically per breed group.
- **Data Augmentation**: Realistic transforms (random rotation, horizontal flip, color jitter) applied only to training images.
- **Model Architectures**: Includes both a scratch baseline (`BaselineCNN`) and transfer learning models (`EfficientNetV2-S`, `ResNet-18`, `MobileNetV3-Small`).
- **Evaluation Metrics**: Test accuracy, precision, recall, macro F1-score, per-class metrics, confusion matrix, and training/validation loss & accuracy curves.

---

## Directory Structure

```text
Demo_2_breed_model/
├── dataset.py                # Dataset preparation, split generation, and PyTorch DataLoaders
├── models.py                 # Custom CNN baseline, EfficientNetV2-S, and ResNet-18 definitions
├── train.py                  # CLI training script with checkpointing and loss/accuracy curves
├── evaluate.py               # Test set evaluation, metrics summary, and confusion matrix plotting
├── demo_notebook.ipynb       # Interactive, self-contained walkthrough notebook
├── split_manifest.json       # Generated reproducible train/val/test file manifest
├── best_model.pth            # Saved PyTorch checkpoint of best-performing model
├── history.json              # Epoch-by-epoch training and validation loss/accuracy history
├── training_curves.png       # Training vs Validation loss and accuracy curves
├── confusion_matrix.png      # Confusion matrix on held-out test split
├── sample_predictions.png    # Visual error analysis showing correct vs incorrect predictions
└── evaluation_metrics.json   # Machine-readable test evaluation metrics
```

---

## Quick Start (Running via CLI)

### 1. Data Preparation & Split Verification
Generate the 70/15/15 split manifest and verify DataLoader:
```powershell
python Demo_2_breed_model/dataset.py
```

### 2. Train a Model
Train using Transfer Learning (EfficientNetV2 or ResNet-18) or Baseline CNN:

```powershell
# Pretrained EfficientNetV2-S (Rubric: EfficientNetV2 family)
python Demo_2_breed_model/train.py --model efficientnet_v2 --epochs 5 --batch_size 16 --lr 0.0003

# Pretrained ResNet-18 (Rubric: ResNet family)
python Demo_2_breed_model/train.py --model resnet18 --epochs 5 --batch_size 16 --lr 0.0003

# Custom 3-Block Baseline CNN (trained from scratch)
python Demo_2_breed_model/train.py --model baseline --epochs 10 --batch_size 16 --lr 0.001
```

Training automatically saves:
- `best_model.pth`: Checkpoint with the highest validation accuracy.
- `history.json`: Epoch logs.
- `training_curves.png`: Loss and accuracy visualization.

### 3. Evaluate on the Test Set
Evaluate the saved checkpoint against the held-out test split:
```powershell
python Demo_2_breed_model/evaluate.py --checkpoint Demo_2_breed_model/best_model.pth
```
This prints the metrics table and generates:
- `confusion_matrix.png`
- `sample_predictions.png`
- `evaluation_metrics.json`

---

## How to Scale This Template for the Group Project

The final ITRI626 submission requires comparing **3 model families** across the group:
- **`JP_Model/`**
- **`Lindani_Model/`**
- **`Sulaiman_Model/`**

### Steps to Scale:
1. **Extend Breeds in `dataset.py`**:
   Replace `DEFAULT_BREEDS` with the target list of breeds or all 120 breeds from `archive/images/Images`.
2. **Assigned Architecture Families**:
   - **Lindani** (`Lindani_Model/`): Small Custom Baseline CNN (from scratch benchmark).
   - **JP** (`JP_Model/`): EfficientNetV2 (Modern compound-scaled CNN with transfer learning).
   - **Sulaiman** (`Sulaiman_Model/`): Swin Transformer (`swin_t` / `swin_s` hierarchical vision transformer with transfer learning).
3. **Keep the Shared Split**:
   Share the generated `split_manifest.json` across all three models so that all team members evaluate on the **exact same test set**, as strictly required by Rubric 7.1.
