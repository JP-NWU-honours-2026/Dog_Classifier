# ITRI626 Mini-Project: Comparative Performance Benchmark

This benchmark evaluates the baseline CNN against transfer learning architectures on both standard and curated dataset splits, directly reporting the 4 required academic dimensions.

## 1. The 4 Core Comparison Dimensions

| Model Evaluation | Architecture | Pretrained? | Trainable Parameters | Top-1 Test Acc | Top-5 Test Acc | Macro F1 | Training Time | GPU Hardware |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **JP Baseline** | Custom 5-Block CNN | No (Scratch) | **4,000,000** | **53.88%** | **84.78%** | **0.5320** | **64.0 min** | Tesla T4 GPU |
| **Lindani Baseline** | Custom 5-Block CNN | No (Scratch) | 4,100,000 | 50.50% | 81.20% | 0.4980 | 70.0 min | Tesla T4 GPU |
| **JP Run 1 (Unclean)** | EfficientNetV2-S | Yes (ImageNet) | **20,331,208** | **83.12%** | **97.75%** | **0.8254** | **14.74 min** | NVIDIA GeForce RTX 4050 Laptop GPU |
| **JP Run 2 (Clean)** | EfficientNetV2-S | Yes (ImageNet) | **20,331,208** | **83.25%** | **98.07%** | **0.8268** | **14.20 min** | NVIDIA GeForce RTX 4050 Laptop GPU |

---

## 2. Cross-Evaluation & Data Leakage Impact

To scientifically measure the effect of dataset contamination and cross-split leakage (74 train-val/test duplicate leaks identified in `split_manifest.json`):

| Evaluation Scenario | Test Manifest | Test Size | Top-1 Accuracy | Top-5 Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Model A on Original Test Split** | `split_manifest.json` (with leaks) | 3,200 | 83.12% | 97.75% | 0.8254 |
| **Model A Cross-Eval on Clean Test** | `split_manifest_clean.json` (leaks removed) | 3,164 | 83.25% | 97.72% | 0.8266 |
| **Model B on Clean Test Split** | `split_manifest_clean.json` (clean training) | 3,164 | **83.25%** | **98.07%** | **0.8268** |

---

## 3. Key Scientific Insights for Project Report (Sections 6 & 7)

1. **Transfer Learning vs. From-Scratch CNN**:
   - EfficientNetV2-S pretrained on ImageNet achieves **83.25% Top-1 Test Accuracy** in just **14.2 minutes** (5 epochs), outperforming the from-scratch baseline (**53.88%** after 64 minutes / 60 epochs) by **+29.37 percentage points** while training in **under a quarter of the time** (**14.2 min** vs **64.0 min**).
2. **Top-5 Accuracy Jump**:
   - Top-5 accuracy increases from **84.78%** (Scratch) to **98.07%** (Clean Transfer Learning), confirming that fine-grained breed confusion is drastically mitigated by pre-learned hierarchical representations.
3. **Data Quality & Deduplication Effect**:
   - Model B (Clean) achieves the highest Top-5 Accuracy (**98.07%** vs **97.75%**, a **+0.32% gain**) and highest Macro F1 (**0.8268** vs **0.8254**), while eliminating all false memorization risks caused by identical images leaking between splits.
   - Removing 170 duplicates and conflicting labels reduced training duration from **14.74 min** down to **14.20 min** without loss of class representation across all 120 breeds.
