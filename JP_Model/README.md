# JP_Model - 120-Breed EfficientNetV2 Dog Classifier

This folder contains the complete, reproducible image classification pipeline for **JP's model**, fine-tuning **EfficientNetV2-S** on **all 120 dog breeds** from the Stanford Dogs Dataset for the **ITRI626 Deep Learning Project**.

---

## Architecture & Specifications
- **Model**: `EfficientNetV2-S` (`torchvision.models.efficientnet_v2_s`)
- **Pretrained Weights**: `ImageNet-1K` (`EfficientNet_V2_S_Weights.DEFAULT`)
- **Total Parameters**: 20.3 Million (Classifier head adapted to 120 classes)
- **Input Resolution**: $224 \times 224 \times 3$
- **Loss Function**: `nn.CrossEntropyLoss(label_smoothing=0.1)`
- **Optimizer**: `AdamW(lr=0.0003, weight_decay=0.01)` with `CosineAnnealingLR`
- **Hardware**: Accelerated on **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)** via PyTorch CUDA & AMP (Mixed Precision).

---

## Directory Structure

```text
JP_Model/
├── dataset.py                # 120-breed dataset scanner, 70/15/15 split, PyTorch DataLoaders
├── model.py                  # Pretrained EfficientNetV2-S definition (120 classes)
├── train.py                  # Training pipeline with GPU mixed precision & checkpointing
├── evaluate.py               # Test set evaluation: Top-1, Top-5, Macro F1, Confusion Matrix
├── jp_model_notebook.ipynb   # Interactive Jupyter notebook with visible outputs & analysis
├── split_manifest.json       # Generated 70/15/15 split manifest across all 120 breeds
├── best_model.pth            # Saved PyTorch checkpoint of best-performing epoch
├── history.json              # Epoch-by-epoch loss & accuracy logs
├── training_curves.png       # Training vs Validation loss & accuracy curves
├── confusion_matrix.png      # 120-class test set confusion matrix heatmap
├── sample_predictions.png    # Qualitative predictions & error analysis grid
├── evaluation_metrics.json   # Machine-readable performance metrics
└── README.md                 # This file
```

---

## Quickstart Instructions

### 1. Data Preparation & Split Verification
Index all 120 breeds (~20,580 images) and generate the reproducible 70/15/15 split:
```powershell
python JP_Model/dataset.py
```

### 2. Fast Pipeline Verification (Quick Test)
Run a 30-second verification run on a 5% data subset to confirm everything works end-to-end:
```powershell
python JP_Model/train.py --quick_test
```

### 3. Full Model Training
Train on the full 120-breed dataset across all 20,580 images:
```powershell
# On RTX 4050 GPU (batch size 32, ~2 minutes per epoch):
python JP_Model/train.py --epochs 5 --batch_size 32 --lr 0.0003

# On CPU (batch size 16):
python JP_Model/train.py --epochs 5 --batch_size 16 --lr 0.0003
```

This automatically generates:
- `best_model.pth`: Peak validation accuracy weights.
- `history.json`: Metrics across all epochs.
- `training_curves.png`: Loss and accuracy curves.

### 4. Evaluate on Held-Out Test Set
Evaluate the trained checkpoint on the 15% held-out test split (~3,087 images):
```powershell
python JP_Model/evaluate.py --checkpoint JP_Model/best_model.pth
```
This prints the metrics table and generates:
- `confusion_matrix.png`
- `sample_predictions.png`
- `evaluation_metrics.json`
