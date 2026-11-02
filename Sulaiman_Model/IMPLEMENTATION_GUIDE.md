# ITRI626 Implementation Guide: Swin Transformer (Swin-T)

**Student:** Sulaiman
**Model family:** Swin Transformer (hierarchical vision transformer, transfer learning)
**Folder:** `Dog_Classifier/Sulaiman_Model`
**Pipeline:** JP's (`JP_Model/`), with Swin-T in place of EfficientNetV2-S

This guide explains how the Swin-T part of the project is built and how to
run it. Results and settings are summarised in `README.md`; draft report text
is in `REPORT_SECTIONS.md`.

---

## 1. Role in the group project

The three models are three different families, as the brief requires:

| Member | Model | Family | Pretrained |
|---|---|---|---|
| Lindani | Custom 5-block CNN | CNN, from scratch | no |
| JP | EfficientNetV2-S | modern CNN | ImageNet-1K |
| Sulaiman | Swin-T | vision transformer | ImageNet-1K |

To make the two pretrained models directly comparable, Swin-T is trained with
JP's code, recipe and split files. The only difference between the
EfficientNetV2-S and Swin-T runs is the architecture.

---

## 2. How the code is put together

All four scripts are copies of JP's, changed only where Swin-T needs it:

| File | Taken from | What changed |
|---|---|---|
| `dataset.py` | `JP_Model/dataset.py` | JP's split files store paths from his laptop; a path that does not exist on this machine is redirected to `archive/images/Images/<breed>/<file>` |
| `model.py` | `JP_Model/model.py` | builds `swin_t` instead of `efficientnet_v2_s`; `freeze_backbone` / `unfreeze_backbone` act on Swin's `features`, `norm` and `head` |
| `train.py` | `JP_Model/train.py` | model names in the logs; `5%` written as `5%%` in one help text (Python 3.14 rejects a bare `%`) |
| `evaluate.py` | `JP_Model/evaluate.py` | printed labels; also writes `predictions.csv` (one row per test photo) for the shared scorer |

`compare_all.py` produces the group comparison (section 4b).
`make_architecture_figure.py` draws `swin_architecture.png`. Neither is part
of training.

### The model

`get_model()` loads `torchvision.models.swin_t` with
`Swin_T_Weights.IMAGENET1K_V1` and replaces the 1,000-class head with
`Dropout(0.3) → Linear(768 → classes)`.

Swin-T's `model.features` has 8 parts:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Part | patch embedding | stage 1 | patch merging | stage 2 | patch merging | stage 3 | patch merging | stage 4 |

- `freeze_backbone()`: everything frozen except the head.
- `unfreeze_backbone(6)`: `features[6:]` (last patch merging + stage 4),
  the final `norm` and the head train; parts 0 to 5 stay frozen. This is
  JP's default `unfreeze_from_stage=6`.

![Swin-T architecture](swin_architecture.png)

### The training recipe (`train.py`, JP's defaults)

| | Phase 1 | Phase 2 |
|---|---|---|
| What trains | head only (92,280 weights) | `features[6:]` + norm + head (15,460,392 weights, 56 %) |
| Epochs | 2 | 3 |
| Optimiser | AdamW, lr 3e-4, weight decay 0.01 | AdamW, lr 1e-5, weight decay 0.01 |
| Schedule | cosine to 3e-5 | cosine to 1e-7 |

Shared settings: cross entropy with label smoothing 0.1, batch 32,
224 × 224 input with ImageNet normalisation, training-only augmentation
(horizontal flip, rotation up to 15°, colour jitter), mixed precision,
seed 42. The checkpoint with the best validation accuracy is kept; there is
no early stopping.

### The data

All split files live in `JP_Model/` and are shared by the whole group:

| File | Classes | Train / val / test | Used for |
|---|---|---|---|
| `split_manifest.json` | 120 | 14,355 / 3,025 / 3,200 | `unclean_transfer` |
| `split_manifest_clean.json` | 120 | 14,257 / 2,989 / 3,164 | clean-test evaluation of the unclean model |
| `split_manifest_simplified.json` | 114 | 14,257 / 2,989 / 3,164 | `simplified_transfer` |

The clean file removes 170 duplicate, leaked or double-labelled photos. The
simplified file also merges four look-alike groups (Staffordshire, Foxhound,
Poodle, Schnauzer). If no `--manifest` is given, `dataset.py` uses
`JP_Model/split_manifest.json`.

---

## 3. Running it

From the repository root (`Dog_Classifier`), with the Stanford Dogs photos in
`archive/images/Images`.

**Setup**

```powershell
pip install -r Sulaiman_Model/requirements.txt
python Sulaiman_Model/model.py          # prints parameter counts per phase, checks a forward pass
```

The first call downloads the Swin-T weights (about 110 MB) once.

**Optional smoke test** (5 % of the data, 1 epoch per phase, a few minutes):

```powershell
python Sulaiman_Model/train.py --run-name smoke --manifest JP_Model/split_manifest.json --quick_test
```

Check that the trainable parameter count jumps from about 92 thousand in
phase 1 to about 15.5 million in phase 2. Delete `results/runs/smoke_quick/`
afterwards.

**Train** (about 9 minutes each on an RTX 4060 Laptop GPU):

```powershell
python Sulaiman_Model/train.py --run-name unclean_transfer    --manifest JP_Model/split_manifest.json
python Sulaiman_Model/train.py --run-name simplified_transfer --manifest JP_Model/split_manifest_simplified.json
```

**Evaluate on the test set** (once per run):

```powershell
python Sulaiman_Model/evaluate.py --run unclean_transfer    --manifest JP_Model/split_manifest.json
python Sulaiman_Model/evaluate.py --run unclean_transfer    --manifest JP_Model/split_manifest_clean.json --eval-name eval_test_clean
python Sulaiman_Model/evaluate.py --run simplified_transfer --manifest JP_Model/split_manifest_simplified.json
```

**Notebook** (reads the saved results, no GPU needed):

```powershell
jupyter nbconvert --to notebook --execute --inplace Sulaiman_Model/swin_notebook.ipynb
```

---

## 4. What each run writes

```
results/runs/<run>/
├── run_config.json        every setting, GPU, versions, training time
├── metrics.csv            one row per epoch: phase, lr, train/val loss and accuracy, val top-5, val macro F1
├── history.json           the same per-epoch numbers as lists
├── training_curves.png    loss and accuracy per epoch, phase boundary marked
├── best_model.pth         best checkpoint (not in git)
└── eval_test/             (and eval_test_clean/ for the unclean run)
    ├── metrics.json           top-1, top-5, macro precision/recall/F1, latency, parameters, training time, GPU
    ├── metrics_summary.txt    the same, readable
    ├── per_class_metrics.csv  precision, recall, F1, support per breed
    ├── confusion_matrix.csv / .png
    └── sample_predictions.png first 12 test photos, correct in green, wrong in red
```

These are exactly the files JP's runs produce, plus `eval_test/predictions.csv`:

```
photo,true_breed,predicted_breed,confidence,correct,true_in_top5,true_label,predicted_label
n02085620-Chihuahua/n02085620_199.jpg,Chihuahua,Chihuahua,0.9871,True,True,0,0
```

---

## 4b. The group comparison (`compare_all.py`)

The comparison table is computed by **one scorer for all three models**:
Lindani's `Lindani_Model/compare_models.py`, which recomputes accuracy,
top-5, macro precision, recall and F1 (with a 95 % interval) from each
model's per-photo `predictions.csv`, and checks that all models were scored
on the same photographs. `compare_all.py` collects the files, adds parameter
counts, training time and GPU from each `run_config.json`, and runs the
scorer twice:

| Comparison | Models | Test photos |
|---|---|---|
| `original/` | baseline CNN, EfficientNetV2-S, Swin-T | 3,200 (120 classes), plus a clean score without JP's 36 removed test photos |
| `simplified/` | EfficientNetV2-S, Swin-T | 3,164 (114 classes) |

It also draws `validation_curves.png` (validation accuracy and loss per epoch
for both pretrained models on both splits). Missing predictions files are
skipped and listed.

```powershell
git fetch
git merge origin/Lindani-branch        # brings in Lindani_Model/compare_models.py; changes only his folder
python Sulaiman_Model/compare_all.py
```

**For JP's runs to appear**, JP's `evaluate.py` needs the same export. In
`evaluate_run()`, add `prediction_rows = []` before the loop; inside the loop,
after the top-5 lines, append one dict per photo (`photo` as
`breed_folder/file_name`, `true_breed`, `predicted_breed`, `confidence`,
`correct`, `true_in_top5`); after the loop, write the list to
`eval_out_dir / "predictions.csv"` with `csv.DictWriter`. The exact lines are
in `Sulaiman_Model/evaluate.py` (search for "predictions.csv"). Then rerun
his three `evaluate.py` commands; no retraining is needed.

---

## 5. Problems already met

| Problem | Cause | Fix |
|---|---|---|
| `ValueError: badly formed help string` | Python 3.14 checks argparse help text; JP's `train.py` has `5%` | already fixed here (`5%%`); JP's copy still has it |
| `FileNotFoundError` for an image under `C:\Users\jpret\...` | JP's split files contain his own paths | already handled in `dataset.py`; check the photos are in `archive/images/Images` |
| Training slower per epoch than expected | JP's loaders use `num_workers=0`, so the GPU waits for image loading | leave as is to keep the comparison identical |
| Plot titles say "120" on the 114-class run | hard-coded in JP's plotting code | cosmetic only |

---

## 6. For the report and presentation

- Results, settings and the comparison with EfficientNetV2-S and the
  baseline CNN: `README.md`, `swin_notebook.ipynb` section 10.
- Draft text for sections 4 to 7, references and likely oral questions:
  `REPORT_SECTIONS.md`.
- Figures: `swin_architecture.png`, and in each run folder
  `training_curves.png`, `eval_test/confusion_matrix.png`,
  `eval_test/sample_predictions.png`.
- Caveat to state: Stanford Dogs was built from ImageNet, which Swin-T was
  pretrained on, so the pretrained models' scores are optimistic.

The earlier Swin-T run (full fine-tuning, Lindani-based pipeline, 85.4 %)
and its original guide, split file and diagram are in
`previous_run_lindani_pipeline/`.
