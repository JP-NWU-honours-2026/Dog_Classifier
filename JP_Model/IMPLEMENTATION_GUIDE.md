# ITRI626 Implementation Guide: EfficientNetV2 Model
**Student Assigned**: JP  
**Model Family**: EfficientNetV2 (Transfer Learning)  
**Location**: `Dog_Classifier/JP_Model`

---

## 1. Project Overview & Your Role

In the ITRI626 group project, your responsibility is to build, fine-tune, and evaluate the **EfficientNetV2** model family using **transfer learning with pretrained ImageNet weights**.

EfficientNetV2 uses progressive learning and neural architecture search to optimize training speed and parameter efficiency. In the final report, your results will demonstrate how a state-of-the-art scaled CNN performs compared to Lindani's Baseline CNN and Sulaiman's ResNet.

---

## 2. Copy-Paste AI Prompt

If you are using an AI assistant to build your code, copy and paste this exact prompt:

```text
I am working on the NWU ITRI626 Deep Learning mini-project in the folder "Dog_Classifier/JP_Model". 
My assigned architecture is EfficientNetV2 using transfer learning with pretrained ImageNet weights in PyTorch.

Requirements:
1. Dataset: Read from "../archive/images/Images". Allow me to easily configure how many dog breeds to classify (e.g., 2, 5, 10, or all 120 breeds).
2. Data Splitting: Create or reuse the reproducible 70% Train / 15% Validation / 15% Test split with random seed = 42, ensuring no data leakage.
3. Augmentations: Apply realistic augmentations (RandomHorizontalFlip, RandomRotation(15), ColorJitter) to the training set only. Resize images to 224x224 and normalize with ImageNet mean/std.
4. Model Architecture: Use torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT). Replace the final classifier layer with a Dropout(0.3) and Linear layer matching the number of target dog breeds.
5. Training Loop: Use CrossEntropyLoss, AdamW optimizer (lr ~ 0.0003, weight_decay=1e-2), ReduceLROnPlateau scheduler, and save the best checkpoint to "best_model.pth". Track training vs validation loss and accuracy per epoch and save "training_curves.png".
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

Inside your `JP_Model/` folder:

```text
JP_Model/
├── IMPLEMENTATION_GUIDE.md   # This guide
├── dataset.py                # Split generation and PyTorch DataLoaders
├── model.py                  # Pretrained EfficientNetV2-S definition
├── train.py                  # Fine-tuning loop with checkpointing
├── evaluate.py               # Test metrics and confusion matrix
├── efficientnet_notebook.ipynb # Interactive Jupyter notebook with visible outputs
├── split_manifest.json       # Shared reproducible split (seed=42)
├── best_model.pth            # Saved weights of best epoch
├── training_curves.png       # Loss & accuracy curves for report
└── confusion_matrix.png      # Test set confusion matrix for report
```

---

## 5. EfficientNetV2 PyTorch Implementation

```python
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def get_efficientnet_v2(num_classes: int = 5, pretrained: bool = True):
    weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    
    # EfficientNetV2 classifier is Sequential(Dropout, Linear)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model
```

---

## 6. Report Checklist for JP (ITRI626 Rubric)

When compiling your sections for the final written report:
- [ ] **Transfer Learning Strategy**: Explain the pretrained weights (`EfficientNet_V2_S_Weights.DEFAULT`), input resolution ($224 \times 224$), and head replacement.
- [ ] **Hyperparameters**: Optimizer (AdamW), learning rate ($\approx 0.0003$), weight decay ($0.01$), batch size ($16$), number of epochs ($5\text{–}8$).
- [ ] **Figures to Provide**:
  - `training_curves.png` (Training vs Validation loss & accuracy).
  - `confusion_matrix.png` on the test split.
- [ ] **Discussion Points**: Compare convergence speed and accuracy against Lindani's Baseline CNN and Sulaiman's ResNet.
