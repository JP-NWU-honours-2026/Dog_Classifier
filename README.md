# Dog Breed Classification on Stanford Dogs

**ITRI626 Deep Learning mini project. North-West University, 2026.**

| Member | Student number | Model |
|---|---|---|
| JP Retief | 38279118 | EfficientNetV2-S |
| Lindani Mdlalose | 46290303 | Custom CNN from scratch, and MobileNetV3-Large as a size-matched control |
| Sulaiman Sulaman | 45427429 | Swin-T |

Four models classify a photograph into one of the 120 breeds in Stanford Dogs.
Three are initialised from ImageNet weights and one is trained from random
initialisation, so that the value of transfer learning is measured rather than
assumed.

## Result

All models trained on the same 70/15/15 split and scored on the same 3,200 test
photographs.

| Model | Parameters | Pretrained | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|---|---|
| Custom CNN | 4,581,784 | No | 53.84 % | 84.78 % | 0.5320 |
| MobileNetV3-Large | 4,355,752 | Yes | 67.53 % | 91.75 % | 0.6712 |
| EfficientNetV2-S | 20,331,208 | Yes | 83.12 % | 97.75 % | 0.8254 |
| Swin-T | 27,611,634 | Yes | **88.03 %** | **99.31 %** | **0.8747** |

The 34.19-point gap between the from-scratch network and the best model splits
into two parts. MobileNetV3-Large is the same size as the custom CNN but
pretrained, so holding size constant attributes **13.69 points to pretraining**
and the remaining **20.50 points to capacity and architecture**. Swin-T is the
recommended model.

## Where each submission item is

| Required item | Location |
|---|---|
| Final report, PDF | `report/` |
| Presentation slides | `report/` |
| Notebooks with visible outputs | `Lindani_Model/kaggle_train.ipynb`, `JP_Model/jp_model_notebook.ipynb`, `Sulaiman_Model/swin_notebook.ipynb` |
| Data preparation and evaluation scripts | `Lindani_Model/`, `JP_Model/`, `Sulaiman_Model/` |
| Setup and reproduction steps | This file, and a README in each model folder |
| Train, validation and test split files | `split_manifest.json`, one identical copy in each model folder, verified by `Lindani_Model/split_check.py` |

## Dataset

Stanford Dogs (Khosla *et al.*, 2011), 20,580 photographs across 120 breeds,
taken from the Kaggle mirror `jessicali9530/stanford-dogs-dataset`.

**The photographs are not in this repository.** They are about 820 MB and are
excluded by `.gitignore`. To obtain them:

```bash
kaggle datasets download -d jessicali9530/stanford-dogs-dataset
```

Unzip so that the breed folders sit at `archive/images/Images/n02085620-Chihuahua/`
and so on, at the repository root. Every script finds them from there, or from the
path in the `DOG_IMAGES_DIR` environment variable.

The dataset authors publish no licence of their own. The operative terms are
ImageNet's, which permit non-commercial research and education only. Full
provenance, licence position and ethics notes are in
`Lindani_Model/DATASET_PROVENANCE.md`.

## Setup

Python 3.11 or later, from the repository root:

```bash
python -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

```bash
pip install torch torchvision pillow imagehash matplotlib numpy scikit-learn
```

A CUDA GPU is needed to train in reasonable time. The custom CNN was trained on a
Kaggle Tesla T4, the three pretrained models on an RTX 4050 Laptop GPU.

## Reproducing the work

### 1. Verify the shared split

All four models read the same `split_manifest.json`, produced with seed 42 and
stratified by breed. One identical copy sits in each model folder. Confirm it
reproduces exactly:

```bash
python Lindani_Model/split_check.py
```

Expect 14,355 training, 3,025 validation and 3,200 test photographs across 120
breeds, and all five checks to pass.

### 2. Check the data for duplicates and label conflicts

```bash
python Lindani_Model/duplicate_check.py
```

Fingerprints every photograph with two independent perceptual hashes. Finds 89
groups of byte-identical files, 173 confirmed duplicate pairs, 34 test
photographs with a duplicate in training, and 50 pairs filed under two different
breeds. Takes about ten minutes the first time, then caches.

### 3. Train and evaluate each model

**Custom CNN, from scratch:**

```bash
python Lindani_Model/train.py --name iter2_deeper --convs-per-block 1,1,1,2,2 --epochs 60 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --dropout 0.3 --patience 5 --seed 42
```

```bash
python Lindani_Model/evaluate.py --run iter2_deeper --final
```

**MobileNetV3-Large, the size-matched control:**

```bash
python Lindani_Model/mobilenet/train.py --epochs 5 --batch_size 32 --lr 0.0003 --unfreeze_stage 14
```

```bash
python Lindani_Model/mobilenet/evaluate.py --checkpoint Lindani_Model/mobilenet/best_model.pth
```

**EfficientNetV2-S:**

```bash
python JP_Model/train.py --epochs 5 --batch_size 32 --lr 0.0003
```

```bash
python JP_Model/evaluate.py --checkpoint JP_Model/best_model.pth
```

**Swin-T:** see `Sulaiman_Model/README.md` for the equivalent two commands.

Each model's `evaluate.py` writes a `predictions.csv`, the per-class metrics, the
confusion matrix and sample prediction grids into its own results folder.

### 4. Build the comparison table

One scorer reads every model's predictions, so the comparison is not three
separate implementations of the same metrics. It also checks that all models were
scored on the same photographs.

```bash
python Lindani_Model/compare_models.py baseline_cnn=<path>/predictions.csv mobilenetv3=<path>/predictions.csv efficientnetv2=<path>/predictions.csv swin_t=<path>/predictions.csv --leaked Lindani_Model/results/duplicate_check/leaked_photos.csv --out Lindani_Model/results/comparison
```

## Layout

```
Dog_Classifier/
├── Lindani_Model/        Custom CNN, MobileNetV3 control, data audits, shared scorer
├── JP_Model/             EfficientNetV2-S
├── Sulaiman_Model/       Swin-T
├── Dog_Classifier_App/   Web interface for running a trained model on a photograph
├── Demo_2_breed_model/   Early two-breed proof of concept
├── report/               Final report and slides
└── archive/              The photographs, not committed
```

## Notes on reproducibility

**Model weights are not committed.** `.pth` checkpoints are 6 to 110 MB each and
are excluded by `.gitignore`. Rerun the training to regenerate them.

**Runs are seeded but not bitwise deterministic on a GPU.** Two runs of the same
configuration under seed 42 gave validation macro F1 of 0.5527 and 0.5649, a
spread of 1.2 points. Differences smaller than that are not treated as evidence
anywhere in the report.

**The test partition is read once.** The custom CNN's training script never opens
it, and its evaluation script refuses a second test scoring without an explicit
override, so the final figure is not a measure of how well the model was tuned
against the test photographs.

## References

> KHOSLA, A., JAYADEVAPRAKASH, N., YAO, B. & FEI-FEI, L. 2011. Novel dataset for
> fine-grained image categorization. Paper presented at the First Workshop on
> Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and
> Pattern Recognition, Colorado Springs, CO.

> DENG, J., DONG, W., SOCHER, R., LI, L.-J., LI, K. & FEI-FEI, L. 2009. ImageNet:
> a large-scale hierarchical image database. Paper presented at the IEEE
> Conference on Computer Vision and Pattern Recognition, Miami, FL.
