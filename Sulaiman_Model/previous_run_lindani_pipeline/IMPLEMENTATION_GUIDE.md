# ITRI626 Implementation Guide: Swin Transformer Model
**Student Assigned**: Sulaiman  
**Model Family**: Swin Transformer (Hierarchical Vision Transformer - Transfer Learning)  
**Location**: `Dog_Classifier/Sulaiman_Model`

---

## 1. Project Overview & Your Role

In the ITRI626 group project, your responsibility is to build, fine-tune, and evaluate the **Swin Transformer** architecture using **transfer learning with pretrained ImageNet weights** (specifically `Swin-T` - Swin Transformer Tiny, or `Swin-S`).

### Why Swin Transformer is a Standout Choice:
* Unlike traditional Vision Transformers (ViT) that have quadratic computational complexity, Swin Transformer builds **hierarchical feature maps** and computes self-attention within **Shifted Windows** with linear complexity.
* Your model provides the group with a **Vision Transformer / Attention-based architecture**, perfectly contrasting Lindani's **Custom Baseline CNN** and JP's **EfficientNetV2 CNN**. This fulfills the ITRI626 rubric requirement for *three meaningfully different model families*.

---

## 2. Copy-Paste AI Prompt

If you are using an AI assistant to build your code, copy and paste this exact prompt:

```text
I am working on the NWU ITRI626 Deep Learning mini-project in the folder "Dog_Classifier/Sulaiman_Model". 
My assigned architecture is the Swin Transformer family (Swin-T / Swin Transformer Tiny) using transfer learning with pretrained ImageNet weights in PyTorch.

Requirements:
1. Dataset: Read from "../archive/images/Images". Allow me to easily configure how many dog breeds to classify (e.g., 2, 5, 10, or all 120 breeds).
2. Data Splitting: Create or reuse the reproducible 70% Train / 15% Validation / 15% Test split with random seed = 42, ensuring no data leakage.
3. Augmentations: Apply realistic augmentations (RandomHorizontalFlip, RandomRotation(15), ColorJitter) to the training set only. Resize images to 224x224 and normalize with ImageNet mean/std.
4. Model Architecture: Use torchvision.models.swin_t(weights=Swin_T_Weights.DEFAULT). Replace the final classification head (model.head) with a Dropout(0.3) and Linear layer matching the number of target dog breeds.
5. Training Loop: Use CrossEntropyLoss, AdamW optimizer (learning rate ~ 0.0001, weight_decay=0.05), cosine or plateau learning rate scheduler, and save the best checkpoint to "best_model.pth". Track training vs validation loss and accuracy per epoch and save "training_curves.png".
6. Evaluation: Evaluate "best_model.pth" on the held-out test set. Output Test Accuracy, Precision, Recall, Macro F1-Score, per-class metrics, and generate "confusion_matrix.png".
7. Deliverables: Provide dataset.py, model.py, train.py, evaluate.py, and a Jupyter Notebook with visible outputs.
```

---

## 3. How to Configure the Number of Dog Breeds

You can configure the number of classes in your `dataset.py` (or notebook) dynamically:

```python
from pathlib import Path

# Option A: Align with group on specific chosen breeds
SELECTED_BREEDS = [
    "n02085620-Chihuahua",
    "n02110185-Siberian_husky",
    "n02099601-golden_retriever",
    "n02106662-German_shepherd",
    "n02108089-boxer"
]

# Option B: Top N breeds with highest image counts
def get_top_n_breeds(dataset_dir: Path, n: int = 5):
    breed_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    sorted_breeds = sorted(breed_dirs, key=lambda d: len(list(d.glob("*.jpg"))), reverse=True)
    return [d.name for d in sorted_breeds[:n]]

# Option C: Scale to all 120 breeds
def get_all_breeds(dataset_dir: Path):
    return sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
```

> [!IMPORTANT]
> **Fair Comparison Requirement (Rubric 7.1)**:
> All three group members must train and evaluate on the **same data split** and **same target breeds**. Use `split_manifest.json` with `seed=42` to guarantee this.

---

## 4. Suggested Directory Structure

Inside your `Sulaiman_Model/` folder:

```text
Sulaiman_Model/
├── IMPLEMENTATION_GUIDE.md   # This guide
├── dataset.py                # Split generation and PyTorch DataLoaders
├── model.py                  # Pretrained Swin-T Transformer definition
├── train.py                  # Fine-tuning loop with checkpointing
├── evaluate.py               # Test metrics and confusion matrix
├── swin_notebook.ipynb       # Interactive Jupyter notebook with visible outputs
├── split_manifest.json       # Shared reproducible split (seed=42)
├── best_model.pth            # Saved weights of best epoch
├── training_curves.png       # Loss & accuracy curves for report
└── confusion_matrix.png      # Test set confusion matrix for report
```

---

## 5. Swin Transformer PyTorch Implementation

```python
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import swin_t, Swin_T_Weights

def get_swin_transformer(num_classes: int = 5, pretrained: bool = True):
    """
    Returns a Swin Transformer Tiny model adapted for dog breed classification.
    Pretrained weights: Swin_T_Weights.DEFAULT (ImageNet-1K)
    Input resolution: 224x224
    """
    weights = Swin_T_Weights.DEFAULT if pretrained else None
    model = swin_t(weights=weights)

    # Swin-T classification head is a Linear layer (in_features=768)
    in_features = model.head.in_features  # 768
    model.head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model
```

---

## 6. Recommended Training Hyperparameters for Swin Transformer

Vision Transformers (including Swin) perform best with **AdamW** and slightly smaller learning rates than CNNs:
* **Optimizer**: `AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)`
* **Scheduler**: `CosineAnnealingLR` or `ReduceLROnPlateau(mode='min', factor=0.5, patience=2)`
* **Batch Size**: `16` (fits comfortably in RAM/VRAM)
* **Epochs**: `5` to `8` epochs (transfer learning converges rapidly)

---

## 7. Report Checklist for Sulaiman (ITRI626 Rubric)

When compiling your sections for the final written report:
- [ ] **Shifted Window Attention Concept**: Explain how Swin Transformer limits self-attention to non-overlapping local windows while allowing cross-window connections via window shifting.
- [ ] **Transfer Learning Strategy**: Pretrained weights (`Swin_T_Weights.DEFAULT`), input size ($224 \times 224$), modified head ($768 \rightarrow \text{num\_classes}$).
- [ ] **Hyperparameters**: Optimizer (AdamW), learning rate ($10^{-4}$), weight decay ($0.05$), batch size ($16$), number of epochs ($5\text{–}8$), random seed ($42$).
- [ ] **Figures to Provide**:
  - `training_curves.png` (Training vs Validation loss & accuracy).
  - `confusion_matrix.png` on the held-out test split.
- [ ] **Comparative Discussion Points**: 
  - Compare how a **Vision Transformer** performs versus Lindani's **Custom Baseline CNN** and JP's **EfficientNetV2**.
  - Discuss attention vs convolution: why Swin Transformer captures global context differently than traditional CNNs.
