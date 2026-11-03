# JP_Model - 120-Breed EfficientNetV2 Dog Classifier

This folder contains the complete, reproducible image classification pipeline for **JP's model**, fine-tuning **EfficientNetV2-S** on **all 120 dog breeds** from the Stanford Dogs Dataset for the **ITRI626 Deep Learning Project**.

---

## Transfer Learning Methodology

The model follows standard two-phase deep learning transfer learning:

1. **Phase 1: Feature Extraction / Head Warmup (Frozen Backbone)**:
   - Pretrained ImageNet convolutional backbone is frozen (`base_model.trainable = False` equivalent: `requires_grad = False` on `model.features`).
   - Only the custom 120-class classifier head (153k parameters) is trained using AdamW at `lr=0.0003`.
   - Protects pretrained feature representations from destructive gradients originating from an uninitialized classification head.

2. **Phase 2: Fine-Tuning (Unfrozen Backbone)**:
   - Backbone layers are unfrozen (`requires_grad = True`), either entirely or for upper stages (`--unfreeze_stage 5` or `6`).
   - Optimizer is re-configured with a significantly lower learning rate (`lr=1e-5`).
   - Adapts high-level feature maps gently to fine-grained dog breed distinguishing marks.

---

## Architecture & Specifications
- **Model**: `EfficientNetV2-S` (`torchvision.models.efficientnet_v2_s`)
- **Pretrained Weights**: `ImageNet-1K` (`EfficientNet_V2_S_Weights.DEFAULT`)
- **Total Parameters**: 20.3 Million (Classifier head adapted to 120 classes)
- **Input Resolution**: $224 \times 224 \times 3$
- **Loss Function**: `nn.CrossEntropyLoss(label_smoothing=0.1)`
- **Optimizer**: `AdamW(weight_decay=0.01)` with `CosineAnnealingLR`
  - Phase 1 LR: `0.0003`
  - Phase 2 LR: `0.00001` (1e-5)
- **Hardware**: Accelerated on **NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM)** via PyTorch CUDA & AMP (Mixed Precision).

---

## Directory Structure

```text
JP_Model/
├── dataset.py                # 120-breed dataset scanner, 70/15/15 split, PyTorch DataLoaders
├── model.py                  # Pretrained EfficientNetV2-S with freeze_backbone() & unfreeze_backbone()
├── train.py                  # Two-phase transfer learning pipeline with AMP & checkpointing
├── evaluate.py               # Test set evaluation: Top-1, Top-5, Macro F1, Confusion Matrix
├── jp_model_notebook.ipynb   # Interactive Jupyter notebook with visible outputs & analysis
├── split_manifest.json       # Generated 70/15/15 split manifest across all 120 breeds
├── best_model.pth            # Saved PyTorch checkpoint of best-performing epoch
├── history.json              # Epoch-by-epoch loss & accuracy logs annotated with phase
├── training_curves.png       # Training vs Validation loss & accuracy curves with phase boundary
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
Run a ~30-second verification run (1 head epoch + 1 fine-tuning epoch on 5% data subset):
```powershell
python JP_Model/train.py --quick_test
```

### 3. Full Two-Phase Model Training
Train on the full 120-breed dataset across all 20,580 images:
```powershell
# Default: 2 head warmup epochs (lr=3e-4) + 3 fine-tuning epochs on upper stages only (lr=1e-5)
# Lower stages 0-5 remain permanently frozen
python JP_Model/train.py --head_epochs 2 --fine_tune_epochs 3 --batch_size 32

# Or resume fine-tuning an existing checkpoint directly without retraining the head:
python JP_Model/train.py --resume JP_Model/best_model.pth --phase2_only --fine_tune_epochs 3
```

> [!NOTE]
> **Base Weights Lifecycle**:
> - Running `train.py` without `--resume` **automatically resets to pristine ImageNet base weights** loaded directly from PyTorch's local hub cache. You never train "from scratch" (random noise) and you never need to manually reset files.
> - To continue fine-tuning an existing run, pass `--resume JP_Model/best_model.pth`.

This automatically generates:
- `best_model.pth`: Peak validation accuracy weights across both phases.
- `history.json`: Metrics across all epochs with phase tags.
- `training_curves.png`: Loss and accuracy curves with phase transition line.

### 4. Evaluate on Held-Out Test Set
Evaluate the trained checkpoint on the 15% held-out test split (~3,200 images):
```powershell
python JP_Model/evaluate.py --checkpoint JP_Model/best_model.pth
```
This prints the metrics table and generates:
- `confusion_matrix.png`
- `sample_predictions.png`
- `evaluation_metrics.json`
