# ITRI626 Implementation Guide: Baseline CNN Model
**Student Assigned**: Lindani  
**Model Family**: Small Custom Convolutional Neural Network (Baseline)  
**Location**: `Dog_Classifier/Lindani_Model`

---

## 1. Project Overview & Your Role

In the ITRI626 group project, your responsibility is to build, train, and evaluate the **Small Custom Convolutional Neural Network Baseline** from scratch (without pretrained weights).

This baseline serves as the **critical comparison benchmark** against JP's EfficientNetV2 and Sulaiman's ResNet. In the final report, your results will demonstrate how much transfer learning improves accuracy over a custom architecture trained from scratch.

---

## 2. Copy-Paste AI Prompt

If you are using an AI assistant to build your code, copy and paste this exact prompt:

```text
I am working on the NWU ITRI626 Deep Learning mini-project in the folder "Dog_Classifier/Lindani_Model". 
My assigned architecture is a custom Baseline Convolutional Neural Network (CNN) built and trained from scratch in PyTorch (no pretrained weights).

Requirements:
1. Dataset: Read from "../archive/images/Images". Allow me to easily configure how many dog breeds to classify (e.g., 2, 5, 10, or all 120 breeds).
2. Data Splitting: Create a reproducible 70% Train / 15% Validation / 15% Test split with random seed = 42, ensuring no data leakage between splits.
3. Augmentations: Apply realistic augmentations (RandomHorizontalFlip, RandomRotation(15), ColorJitter) to the training set only. Resize images to 224x224.
4. Model Architecture: Implement a modular 3-to-4 block CNN with Conv2D, BatchNorm2D, ReLU, MaxPool2D, AdaptiveAvgPool2d, Dropout (0.4), and a Linear classification head matching the number of selected breeds.
5. Training Loop: Use CrossEntropyLoss, Adam or AdamW optimizer (lr ~ 0.001), learning rate scheduler (ReduceLROnPlateau), and save the best checkpoint to "best_model.pth". Track training vs validation loss and accuracy per epoch and save "training_curves.png".
6. Evaluation: Evaluate "best_model.pth" on the held-out test set. Output Test Accuracy, Precision, Recall, Macro F1-Score, per-class metrics, and generate "confusion_matrix.png".
7. Deliverables: Provide dataset.py, model.py, train.py, evaluate.py, and a Jupyter Notebook with visible outputs.
```

---

## 3. How to Configure the Number of Dog Breeds

You can decide whether to train on **2 breeds** (fastest), **5–10 breeds** (recommended for strong report analysis), or **all 120 breeds**.

In your `dataset.py` (or notebook), configure the breed selection like this:

```python
from pathlib import Path

# 1. Option A: Select custom specific breeds
SELECTED_BREEDS = [
    "n02085620-Chihuahua",
    "n02110185-Siberian_husky",
    "n02099601-golden_retriever",
    "n02106662-German_shepherd",
    "n02108089-boxer"
]

# 2. Option B: Automatically pick the top N breeds with the most images
def get_top_n_breeds(dataset_dir: Path, n: int = 5):
    breed_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    # Sort by number of images descending
    sorted_breeds = sorted(breed_dirs, key=lambda d: len(list(d.glob("*.jpg"))), reverse=True)
    return [d.name for d in sorted_breeds[:n]]

# 3. Option C: Classify all 120 breeds
def get_all_breeds(dataset_dir: Path):
    return sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
```

> [!IMPORTANT]
> **Group Alignment**:
> Coordinate with JP and Sulaiman so that all three models evaluate on the **exact same selected breeds** and **identical test split** (`seed=42`)!

---

## 4. Suggested Directory Structure

Inside your `Lindani_Model/` folder, organize your files as follows:

```text
Lindani_Model/
├── IMPLEMENTATION_GUIDE.md   # This guide
├── dataset.py                # Split generation and PyTorch DataLoaders
├── model.py                  # Baseline CNN architecture (from scratch)
├── train.py                  # Training pipeline with checkpointing
├── evaluate.py               # Test metrics and confusion matrix
├── baseline_notebook.ipynb   # Interactive Jupyter notebook with visible outputs
├── split_manifest.json       # Generated split (reproducible seed=42)
├── best_model.pth            # Saved PyTorch weights of best epoch
├── training_curves.png       # Loss & accuracy curves for report
└── confusion_matrix.png      # Test set confusion matrix for report
```

---

## 5. Recommended Baseline CNN Architecture

```python
import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 224 -> 112

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112 -> 56

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56 -> 28
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))
```

---

## 6. Report Checklist for Lindani (ITRI626 Rubric)

When compiling your sections for the final written report:
- [ ] **Architecture Details**: Number of conv layers, kernel size ($3 \times 3$), activation function (ReLU), pooling, dropout rate ($0.4$), total trainable parameters.
- [ ] **Training Settings**: Optimizer (Adam/AdamW), initial learning rate ($\approx 0.001$), batch size ($16$ or $32$), epochs ($10\text{–}15$), random seed ($42$).
- [ ] **Figures to Provide**:
  - `training_curves.png` (Training vs Validation loss & accuracy).
  - `confusion_matrix.png` on the held-out test split.
- [ ] **Discussion Points**: Explain why the scratch CNN takes longer to converge than pretrained models and discuss any signs of overfitting.
