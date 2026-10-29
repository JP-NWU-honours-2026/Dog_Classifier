# Sulaiman_Model: Swin Transformer (Swin-T)

Model family 3 of 3 in the ITRI626 dog breed project: **Swin-T, pretrained on
ImageNet-1K and fine-tuned** on all 120 breeds of Stanford Dogs. The other two
are Lindani's CNN trained from scratch and JP's EfficientNetV2-S.

## Result

| | Validation (3,025) | Test (3,200) | Clean test (3,166) |
|---|---|---|---|
| Accuracy | 86.61 % | **85.38 %** | 85.50 % |
| Top-5 accuracy | 98.88 % | 98.56 % | |
| Macro F1 | 0.8644 | **0.8490** | 0.8500 |

Clean test = the test set without the 34 photos that have a near-copy in
training (found by `Lindani_Model/duplicate_check.py`).

## Files

| File | Step | What it does |
|---|---|---|
| `dataset.py` | 1 | Reads the shared split, preprocessing, training-only augmentation, data loaders |
| `model.py` | 2 | Swin-T with ImageNet weights and a new Dropout(0.3) + Linear(768 to 120) head |
| `train.py` | 3 | Fine-tunes the model, logs every setting, keeps the best epoch |
| `evaluate.py` | 5 | Scores validation, then the test set once; writes metrics, tables and figures |
| `swin_notebook.ipynb` | 6 | Runs the checks and shows every result with outputs visible |
| `split_manifest.json` | | The shared 70/15/15 split, seed 42, identical to the other two models |
| `requirements.txt` | | Library versions used |
| `results/runs/swin_t/` | 4 | Training log (`metrics.csv`, `run_config.json`) and `eval_val/`, `eval_test/` |

`best_model.pth` (110 MB) is not in git. Rerun training to recreate it.

## Settings

| Setting | Value |
|---|---|
| Model | `torchvision.models.swin_t`, weights `Swin_T_Weights.IMAGENET1K_V1`, 27,611,634 parameters, nothing frozen |
| Input | 224 x 224, ImageNet normalisation |
| Augmentation (training only) | horizontal flip, rotation up to 15 degrees, mild colour jitter |
| Optimiser | AdamW, learning rate 1e-4, weight decay 0.05 |
| Schedule | 1 epoch linear warm-up, then cosine decay to 0, per batch |
| Batch size | 32, no gradient accumulation |
| Epochs | up to 15, early stopping on validation macro F1 with patience 4 (cap reached, best epoch 15) |
| Loss | cross entropy, no label smoothing |
| Other | dropout 0.3, gradient clipping 1.0, mixed precision, seed 42 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), 19 minutes, about 71 s per epoch |

## Reproduce

From the repository root (`Dog_Classifier`), with the Stanford Dogs photos in
`archive/images/Images` (or point `DOG_IMAGES_DIR` at them):

```powershell
pip install -r Sulaiman_Model/requirements.txt

python Sulaiman_Model/dataset.py              # self-test of the split and data
python Sulaiman_Model/model.py                # self-test of the model, downloads weights once
python Sulaiman_Model/train.py                # trains, about 20 minutes on an RTX 4060
python Sulaiman_Model/evaluate.py             # scores validation
python Sulaiman_Model/evaluate.py --final     # scores the test set, once

jupyter nbconvert --to notebook --execute --inplace Sulaiman_Model/swin_notebook.ipynb
```

GPU results are repeatable to within small differences, not bit for bit
(`train.py --deterministic` forces the slower fully repeatable mode).
