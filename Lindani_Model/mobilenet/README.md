# MobileNetV3-Large, transfer learning

A fourth model, added so that the effect of ImageNet pretraining can be measured
with network size held constant.

## Why this model exists

The from-scratch CNN in the parent folder reaches 53.84 per cent on the test set.
EfficientNetV2-S reaches 83.12 and Swin-T 85.38. That gap has two possible
explanations and the existing comparison cannot separate them:

1. the pretrained models start from ImageNet weights, or
2. the pretrained models are four to six times larger.

MobileNetV3-Large has **4,355,752** trainable parameters with a 120 class head.
The from-scratch CNN has **4,581,784**. The two are the same size to within five
per cent, so the difference between them isolates pretraining.

## Provenance: what was copied, and what differs

| File | Source |
|---|---|
| `dataset.py` | **Copied byte for byte** from `JP_Model/` at commit `c254375b` |
| `train.py` | **Copied byte for byte** from `JP_Model/` at commit `c254375b` |
| `evaluate.py` | **Copied byte for byte** from `JP_Model/` at commit `c254375b` |
| `split_manifest.json` | Copied from `JP_Model/`, the original 120 class split |
| `split_manifest_simplified.json` | Copied from `JP_Model/`, the 114 class merged split |
| `model.py` | **The only file written here.** |

Even inside `model.py`, the helper functions `freeze_backbone`,
`unfreeze_backbone`, `count_parameters` and `get_parameter_summary` are copied
unchanged from JP's file. They operate on `model.features` and
`model.classifier`, which MobileNetV3 also has, so they needed no modification.
Only `get_model` is new.

This matters for the report: the data handling, augmentation, training loop,
two phase schedule, optimiser, learning rates and evaluation are not merely
similar to the EfficientNetV2-S run, they are the same code.

## Two deliberate differences, both stated

**1. Weights are pinned to `IMAGENET1K_V1`.**

MobileNetV3-Large has two sets of ImageNet weights, and `DEFAULT` resolves to
`IMAGENET1K_V2`, trained with an improved recipe (75.27 against 74.04 per cent
ImageNet top-1). EfficientNetV2-S and Swin-T only have V1, so taking the default
here would give this model a better starting point than the models it is
compared against. V1 is pinned so that all three use the standard recipe.

**2. The fine-tuning depth is matched by parameters, not by block number.**

JP's `--unfreeze_stage 6` on EfficientNetV2-S leaves **74.0 per cent** of
parameters trainable in phase 2. MobileNetV3-Large has 17 feature blocks rather
than 8, so the same number would mean something quite different:

| Starting block | Trainable |
|---|---|
| 13 | 81.8 per cent |
| **14** | **72.0 per cent** |
| 15 | 53.6 per cent |

So **`--unfreeze_stage 14`** is used here, matching the proportion of the
network that is adapted rather than an index that is not comparable between
architectures.

One smaller point: MobileNetV3's classifier contains a 960 to 1280 expansion
layer that carries pretrained weights. It is kept, and only the final linear
layer is replaced, with dropout set to 0.3 to match JP's head. Replacing the
whole classifier, as his code does for EfficientNetV2, would discard those
pretrained parameters.

## Who runs this, and why

**JP runs it on his machine.** The manifests store absolute paths from the
computer that generated them, and `dataset.py` opens those paths directly, so
the photographs resolve only on his machine. Rewriting the paths would work, but
running it where the data already lives keeps the dataset, the code and the
hardware identical to the EfficientNetV2-S runs, which is the point of the
exercise.

The two manifest files in this folder are byte-identical copies of his, included
so the folder runs regardless of which branch is checked out.

### Handover, for JP

Run both commands from inside this folder. A one minute check first:

```bash
python train.py --run-name mobilenet_smoke --manifest split_manifest.json \
    --unfreeze_stage 14 --quick_test
```

Then the two real runs, about 10 to 15 minutes each:

```bash
python train.py --run-name mobilenet_unclean --manifest split_manifest.json \
    --unfreeze_stage 14 --seed 42

python train.py --run-name mobilenet_simplified --manifest split_manifest_simplified.json \
    --unfreeze_stage 14 --seed 42
```

Everything else is left at your defaults, so the recipe matches your own runs:
2 head epochs then 3 fine-tuning epochs, batch 32, head lr 3e-4, fine-tune lr
1e-5, seed 42.

Results are written to `results/runs/<run-name>/`. Please send back the whole
folder for each run, or at minimum `run_config.json`, `metrics.csv`,
`history.json` and the evaluation outputs. `best_model.pth` is not needed and is
excluded from git anyway.

**`--unfreeze_stage 14` is deliberate**, not a typo. Your `6` leaves 74.0 per
cent of EfficientNetV2-S trainable. MobileNetV3-Large has 17 feature blocks
rather than 8, so 6 would unfreeze far more of it. Block 14 leaves 72.0 per
cent, which matches what you fine-tune.

## The commands, for reference

Two runs, matching the conditions JP used.

**Unclean, the original 120 class split:**

```bash
python train.py --run-name mobilenet_unclean --manifest split_manifest.json \
    --unfreeze_stage 14 --seed 42
```

**Simplified, the 114 class merged split:**

```bash
python train.py --run-name mobilenet_simplified --manifest split_manifest_simplified.json \
    --unfreeze_stage 14 --seed 42
```

Everything else is left at JP's defaults: 2 head epochs then 3 fine-tuning
epochs, batch size 32, head learning rate 3e-4, fine-tuning learning rate 1e-5,
seed 42.

Then evaluate each run with the copied `evaluate.py`, and put the predictions
through `../compare_models.py` alongside the other models.

## A caution about the simplified split

It has **114 classes, not 120**, because it merges the Staffordshire terrier,
foxhound, poodle and schnauzer groups. Accuracy on 114 classes is not directly
comparable with accuracy on 120: merging removes the hardest decisions, so the
number rises for reasons that have nothing to do with the model. Report the two
conditions separately, with the class count stated in both cases.

## Expected cost

Five epochs on a small pretrained network. Roughly 10 to 15 minutes per run on a
Kaggle T4, so both conditions inside half an hour.
