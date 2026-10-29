# Baseline CNN, trained from scratch

Lindani Mdlalose. ITRI626 group project, North-West University.

This folder holds the **control** in the three model comparison: a small
convolutional network trained from random initialisation, with no pretrained
weights. It is the only one of the three models that has never seen a photograph
outside the training split, which is what makes the other two models' use of
transfer learning measurable rather than merely asserted.

## Result

| | |
|---|---|
| **Test accuracy** | **53.84 per cent** (95 per cent interval 52.11 to 55.57) |
| Top-5 accuracy | 84.78 per cent |
| Macro F1 | 0.5320 |
| Chance level | 0.83 per cent on 120 breeds |
| Trainable parameters | 4,581,784 |
| Training | 64 minutes on a Tesla T4 |

The test set was read **once**, after the final model had been selected on
validation. Five configurations were trained; the selection and the reasoning
are in `MODEL_DESIGN.md` and section 4 of `REPORT_DRAFT.md`.

## The dataset

Stanford Dogs (Khosla *et al.*, 2011), 20,580 photographs across 120 breeds,
obtained from the Kaggle mirror `jessicali9530/stanford-dogs-dataset`. The
photographs are not committed; `archive/` at the repository root holds them
locally.

The dataset authors publish no licence. The operative terms are ImageNet's,
which restrict use to non-commercial research and education. Full provenance,
licence position, citations and ethics notes are in `DATASET_PROVENANCE.md`.

The shared 70/15/15 split (`split_manifest.json`, seed 42) was produced by a
group member and adopted unchanged, so that all three models train and are
scored on identical data.

## Setup

Python 3.11 or later. From the repository root:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pillow imagehash matplotlib numpy
pip install torch torchvision          # needed from step 5 onwards
```

Training itself runs on Kaggle, not on a laptop.

## Reproducing the work, in order

Every step writes its output to `results/` and prints a summary. Steps 1 to 4
concern the data and need no GPU.

### 1. Audit the shared split

```bash
python Lindani_Model/split_check.py
```

Checks that the split reproduces exactly under seed 42, is stratified across all
120 breeds, has no photograph in two partitions, and has labels matching the
breed folders. All five checks pass. Output: `results/split_check/`.

### 2. Find near-duplicate photographs

```bash
python Lindani_Model/duplicate_check.py
```

Fingerprints all 20,580 photographs with two independent perceptual hashes and
compares every pair. Takes about ten minutes the first time, then caches.

Finds 89 groups of byte-identical files, 173 confirmed duplicate pairs under the
adopted rule, **34 test photographs with a duplicate in training**, and **50
pairs carrying two different breed labels**. Output:
`results/duplicate_check/`, including `leaked_photos.csv`, which is the
exclusion list used for the clean test score.

### 3. Describe the dataset

```bash
python Lindani_Model/dataset_figures.py
```

Produces the figures and tables for the dataset section: counts per breed, one
photograph from each of the 120 breeds, image dimensions against the 224 pixel
input, and where the dog sits in the frame. Output: `results/dataset_figures/`.

### 4. Audit the labels

```bash
python Lindani_Model/label_audit.py
```

Draws a fixed sample of 100 photographs, renders them onto numbered sheets, and
scores the verdicts recorded in `label_audit_tally.csv` with a Wilson confidence
interval. Run it once to produce the sheets, fill in the verdicts by eye, then
run it again to score. Output: `results/label_audit/`.

### 5. Check the model builds

```bash
python Lindani_Model/model.py
```

Prints the architecture, the parameter count for both variants, and the size of
the data after every block. Expect 224, 112, 56, 28, 14, 7.

### 6. Check the data pipeline

```bash
python Lindani_Model/dataset.py
```

Self-test. Must report 14,355 training, 3,025 validation and 3,200 test
photographs across 120 breeds, and pass its label check.

### 7. Train

Training runs on Kaggle. Open `kaggle_train.ipynb` there, attach the Stanford
Dogs dataset, enable the GPU and internet, then use **Save Version, Run All**.

The final model is reproduced by:

```bash
python Lindani_Model/train.py --name iter2_deeper --convs-per-block 1,1,1,2,2 \
    --epochs 60 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --dropout 0.3 \
    --patience 5 --seed 42
```

Every setting, library version and the GPU used are written automatically to
`results/runs/<name>/run_config.json`.

**The training script never reads the test partition.**

### 8. Evaluate

```bash
python Lindani_Model/evaluate.py --run iter2_deeper              # validation
python Lindani_Model/evaluate.py --run iter2_deeper --final      # test, once
```

Without `--final` it scores validation, which may be repeated freely. With
`--final` it scores the test set, and refuses to do so twice without an explicit
override. This is deliberate: iterating against test scores would quietly turn
the final figure into a measure of how well the model was tuned to the test
photographs.

Produces accuracy with a confidence interval, top-5, macro and weighted
precision, recall and F1, per-breed metrics, the confusion matrix, the top
confusions, learning curves, and grids of correct and incorrect predictions. With
`--final` it also reports the clean test score with the leaked duplicates
excluded.

### 9. Compare all three models

```bash
python Lindani_Model/compare_models.py \
    baseline_cnn=Lindani_Model/results/runs/iter2_deeper/eval_test/predictions.csv \
    efficientnetv2=<jp's predictions.csv> \
    swin_t=<sulaiman's predictions.csv> \
    --leaked Lindani_Model/results/duplicate_check/leaked_photos.csv \
    --out Lindani_Model/results/comparison
```

Scores every model from its own predictions with one piece of code, so the
comparison table is not three different implementations of the same metrics. It
also verifies that all models were scored on the same photographs and warns when
they were not. Standard library only, so it runs without this environment.

## What is in this folder

| File | Purpose |
|---|---|
| `split_check.py` | Audits the shared split |
| `duplicate_check.py` | Finds near-duplicates and label conflicts |
| `dataset_figures.py` | Dataset figures and tables |
| `label_audit.py` | Sample audit of the labels |
| `model.py` | The network, with the architecture variant flag |
| `dataset.py` | Reads the shared split, resolves paths, knows the leaked list |
| `train.py` | Training loop, logging every reproducibility detail |
| `evaluate.py` | Scoring, with the test set gated behind `--final` |
| `compare_models.py` | One scorer for all three models |
| `make_report_docx.py` | Turns the report draft into a Word document |
| `kaggle_train.ipynb` | The Kaggle notebook that ran the training |
| `MODEL_DESIGN.md` | The architecture and why each choice was made |
| `DATASET_PROVENANCE.md` | Source, licence, citations, ethics |
| `REPORT_DRAFT.md` | Draft of the report sections, with every figure sourced |
| `split_manifest.json` | The shared split, as adopted |

Results live under `results/`, one folder per concern, with `results/runs/`
holding the five training runs.

## Notes

**Model weights are not committed.** `.pth` files are 6 to 18 MB each and are
excluded by `.gitignore`. They are regenerated by rerunning the training, or kept
locally for evaluation.

**Runs are seeded but not bitwise deterministic on a GPU.** Two executions of the
same configuration under seed 42 gave validation macro F1 of 0.5527 and 0.5649, a
spread of 1.2 points. Differences smaller than that are not evidence of anything,
and all comparisons in the report are read against that floor.

## Citation

> KHOSLA, A., JAYADEVAPRAKASH, N., YAO, B. & FEI-FEI, L. 2011. Novel dataset for
> fine-grained image categorization. Paper presented at the First Workshop on
> Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and
> Pattern Recognition, Colorado Springs, CO.

> DENG, J., DONG, W., SOCHER, R., LI, L.-J., LI, K. & FEI-FEI, L. 2009. ImageNet:
> a large-scale hierarchical image database. Paper presented at the IEEE
> Conference on Computer Vision and Pattern Recognition, Miami, FL.
