# Sulaiman_Model: Swin Transformer (Swin-T)

Model family 3 of 3 in the ITRI626 dog breed project: **Swin-T, pretrained on
ImageNet-1K, fine-tuned with JP's two-phase transfer learning pipeline**. The
other two models are Lindani's CNN trained from scratch and JP's
EfficientNetV2-S.

The code here is JP's pipeline (`JP_Model/`) with Swin-T swapped in for
EfficientNetV2-S, so both pretrained models are trained, logged and evaluated
by the same code on the same split files.

## Results

| Run | Split file | Classes | Test photos | Top-1 | Top-5 | Macro F1 | Train time |
|---|---|---|---|---|---|---|---|
| `unclean_transfer` | `JP_Model/split_manifest.json` | 120 | 3,200 | **88.03 %** | 99.31 % | **0.8747** | 9.1 min |
| `unclean_transfer` (clean test) | `JP_Model/split_manifest_clean.json` | 120 | 3,164 | 88.21 % | 99.30 % | 0.8764 | |
| `simplified_transfer` | `JP_Model/split_manifest_simplified.json` | 114 | 3,164 | **89.92 %** | 99.34 % | **0.8953** | 9.1 min |

For comparison, with the same pipeline and split files, JP's EfficientNetV2-S
scores 83.12 % / 0.8254 (unclean) and 84.54 % / 0.8402 (simplified).

- **Unclean:** the dataset as downloaded.
- **Clean test:** the unclean model scored on the cleaned test set (36
  duplicate, leaked or double-labelled photos removed).
- **Simplified:** trained and tested on JP's cleaned split with four
  look-alike groups merged (Staffordshire Terrier, Foxhound, Poodle,
  Schnauzer).

## Files

| File | What it does |
|---|---|
| `dataset.py` | JP's data loader. Only change: JP's split files contain paths from his laptop, so missing paths are redirected to `archive/images/Images/<breed>/<file>` |
| `model.py` | JP's model file with Swin-T: `get_model`, `freeze_backbone`, `unfreeze_backbone`, `get_parameter_summary` |
| `train.py` | JP's two-phase trainer, model labels changed only |
| `evaluate.py` | JP's evaluator; printed labels changed, and it also writes `predictions.csv` (one row per test photo) |
| `compare_all.py` | The group comparison: runs Lindani's shared scorer (`Lindani_Model/compare_models.py`) on every model's `predictions.csv`, plus validation curves of both pretrained models |
| `swin_notebook.ipynb` | Shows every setting, training curve, test metric, confusion matrix, per-breed table and comparison, outputs visible |
| `REPORT_SECTIONS.md` | Draft Swin-T text for the group report |
| `IMPLEMENTATION_GUIDE.md` | How the code is put together, how to run it, problems already met |
| `swin_architecture.png` | Architecture diagram for the report: what is frozen and what trains in each phase |
| `make_architecture_figure.py` | Draws `swin_architecture.png` |
| `requirements.txt` | Library versions |
| `results/runs/<run>/` | `run_config.json`, `metrics.csv`, `history.json`, `training_curves.png`, `best_model.pth`, `eval_test/` |
| `results/comparison/` | Output of `compare_all.py`: `original/` and `simplified/` comparison tables, per-model metrics, per-breed and confusion files, `validation_curves.png` |
| `previous_run_lindani_pipeline/` | The earlier 85.4 % run with its own code, split file, guide, diagram and docs, kept for reference (see its `ABOUT.md`) |

The split files are not copied here: all three runs read them from
`JP_Model/` (`split_manifest.json`, `split_manifest_clean.json`,
`split_manifest_simplified.json`), which is also the default when no
`--manifest` is given.

Each `eval_test/` folder holds `metrics.json`, `metrics_summary.txt`,
`per_class_metrics.csv`, `confusion_matrix.csv`, `confusion_matrix.png` and
`sample_predictions.png`, in exactly JP's format, plus `predictions.csv` for
the shared scorer.

`best_model.pth` (about 110 MB) is not in git (`*.pth` is ignored). Rerun
training to recreate it.

## Training recipe (identical to JP's)

| Setting | Value |
|---|---|
| Model | `torchvision.models.swin_t`, weights `Swin_T_Weights.IMAGENET1K_V1`, new head Dropout(0.3) + Linear(768 → classes), 27.6 M parameters |
| Phase 1 | 2 epochs, only the head trains (92 k parameters), AdamW lr 3e-4, cosine to 3e-5 |
| Phase 2 | 3 epochs, `features[6:]` (last patch merging + stage 4) + final norm + head train (15.5 M parameters), AdamW lr 1e-5, cosine to 1e-7 |
| Loss | cross entropy, label smoothing 0.1 |
| Weight decay | 0.01 |
| Batch / input | 32 / 224 × 224, ImageNet normalisation |
| Augmentation (train only) | horizontal flip, rotation up to 15°, colour jitter (brightness 0.15, contrast 0.15, saturation 0.1) |
| Best epoch | highest validation accuracy, no early stopping |
| Other | mixed precision, seed 42 |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), PyTorch 2.11.0+cu128, CUDA 12.8, Python 3.14 |

## Reproduce

From the repository root (`Dog_Classifier`), with the Stanford Dogs photos in
`archive/images/Images`:

```powershell
pip install -r Sulaiman_Model/requirements.txt

python Sulaiman_Model/train.py --run-name unclean_transfer    --manifest JP_Model/split_manifest.json
python Sulaiman_Model/train.py --run-name simplified_transfer --manifest JP_Model/split_manifest_simplified.json

python Sulaiman_Model/evaluate.py --run unclean_transfer    --manifest JP_Model/split_manifest.json
python Sulaiman_Model/evaluate.py --run unclean_transfer    --manifest JP_Model/split_manifest_clean.json --eval-name eval_test_clean
python Sulaiman_Model/evaluate.py --run simplified_transfer --manifest JP_Model/split_manifest_simplified.json

jupyter nbconvert --to notebook --execute --inplace Sulaiman_Model/swin_notebook.ipynb

python Sulaiman_Model/compare_all.py      # needs Lindani_Model/compare_models.py (Lindani's branch)
```

`compare_all.py` scores every model whose `predictions.csv` exists and lists
the ones it skipped. It checks that all models were scored on the same
photographs, and reports a clean score without the 36 test photos removed by
JP's cleaning (the same definition as `split_manifest_clean.json`). It only
reads the other members' folders and writes to `results/comparison/`.

Each training run takes about 9 minutes on an RTX 4060 Laptop GPU. The
first run downloads the Swin-T weights (about 110 MB) once. Add
`--quick_test` to a training command for a 5 % data smoke test.

GPU training is repeatable to within small differences, not bit for bit.

## Notes

- **ImageNet overlap.** Stanford Dogs was built from ImageNet, and Swin-T was
  pretrained on ImageNet-1K, so the backbone has very likely seen many of the
  test photos. Scores of all pretrained models are optimistic for that
  reason.
- **Latency** in `metrics.json` is measured JP's way (no GPU synchronisation),
  so it understates real per-image time. It is comparable with JP's numbers,
  not an absolute figure.
- The plot titles in `training_curves.png` and `confusion_matrix.png` say
  "120" even for the 114-class run; that text is hard-coded in JP's scripts.
