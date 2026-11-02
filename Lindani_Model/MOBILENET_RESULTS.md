# MobileNetV3-Large: results and what they settle

Trained by Jean-Pierre on an RTX 4050 Laptop GPU, 29 October 2026, using the
pipeline copied byte for byte from `JP_Model` at commit `c254375b`. Only the
model definition differed.

## 1. The runs

| | Unclean | Simplified |
|---|---|---|
| Classes | 120 | 114 |
| Test photographs | 3,200 | 3,164 |
| Trainable parameters | 4,355,752 | 4,348,066 |
| **Top-1 accuracy** | **67.53 %** | **68.74 %** |
| Top-5 accuracy | 91.75 % | 92.48 % |
| Macro F1 | 0.6712 | 0.6841 |
| Macro precision | 68.89 % | 70.04 % |
| Macro recall | 67.04 % | 68.33 % |
| Training time | 13.87 min | 13.01 min |
| Inference latency | 0.57 ms per image | 0.57 ms per image |

Settings were JP's defaults throughout: 2 head epochs then 3 fine-tuning epochs,
head learning rate 3e-4, fine-tuning 1e-5, batch 32, label smoothing 0.1, weight
decay 0.01, seed 42, and `--unfreeze_stage 14` as specified so that 72.0 per cent
of parameters were adapted in phase 2, matching the 74.0 per cent adapted in
EfficientNetV2-S.

## 2. The comparison this run existed to make

Both models below were trained on the same 120 class split and scored on the
same 3,200 test photographs.

| | Parameters | Pretrained | Top-1 | Top-5 | Macro F1 | Epochs | Time |
|---|---|---|---|---|---|---|---|
| Baseline CNN | 4,581,784 | **No** | 53.84 % | 84.78 % | 0.5320 | 59 | 64.7 min |
| MobileNetV3-Large | 4,355,752 | **Yes** | **67.53 %** | 91.75 % | **0.6712** | 5 | 13.9 min |
| **Difference** | **5 % fewer** | | **+13.69** | **+6.97** | **+0.139** | | **a fifth of the time** |

**Pretraining is worth 13.69 percentage points at constant network size.** The
pretrained model is in fact slightly smaller, trained for a twelfth of the epochs
and a fifth of the wall clock time, and still beat the from-scratch network by a
wide margin.

If anything that understates it: the from-scratch model was given 59 epochs with
early stopping and a tuned learning rate schedule, while the pretrained one took
five epochs on JP's defaults with no tuning at all.

## 3. What it corrects

The report draft previously assumed that most of the gap between the
from-scratch network and the large pretrained models was pretraining, and
guessed that size was worth "roughly 7 points". **That guess was wrong, and the
measurement is more interesting than the assumption.**

All four models, unclean split, 120 classes, same 3,200 test photographs:

| Model | Parameters | Pretrained | Top-1 |
|---|---|---|---|
| Baseline CNN | 4,581,784 | No | 53.84 % |
| MobileNetV3-Large | 4,355,752 | Yes | 67.53 % |
| EfficientNetV2-S | 20,331,208 | Yes | 83.12 % |
| Swin-T | 27,611,634 | Yes | 88.03 % |

The total gap from the from-scratch baseline to the best model is **34.19
percentage points**, and it now decomposes:

| Component | Points | Share |
|---|---|---|
| **Pretraining**, measured at constant size | **+13.69** | 40 % |
| **Capacity and architecture**, 4.4M to 27.6M parameters | **+20.50** | 60 % |

So pretraining accounts for roughly **two fifths** of the difference, not all of
it. The remaining three fifths comes from the pretrained models also being four
to six times larger and of more modern design.

**One caveat to state plainly:** capacity and architecture cannot be separated
from each other here. EfficientNetV2 and Swin are not merely bigger than
MobileNetV3, they are later designs with different building blocks. Isolating
those two would need a size-matched comparison within the pretrained group,
which this project did not run.

## 4. What the errors show

Weakest and strongest breeds on the unclean split:

| Weakest | F1 | Strongest | F1 |
|---|---|---|---|
| Rhodesian ridgeback | 0.286 | Keeshond | 0.960 |
| Miniature poodle | 0.294 | Komondor | 0.941 |
| Collie | 0.316 | African hunting dog | 0.939 |
| Lhasa | 0.392 | Dandie Dinmont | 0.906 |
| Redbone | 0.394 | Mexican hairless | 0.894 |

Most frequent confusions:

| True breed | Predicted | Times |
|---|---|---|
| French bulldog | Boston bull | 11 |
| Rhodesian ridgeback | Redbone | 11 |
| English foxhound | Walker hound | 9 |
| Siberian husky | Eskimo dog | 8 |
| Eskimo dog | Siberian husky | 8 |
| Lhasa | Shih-Tzu | 8 |
| Appenzeller | Entlebucher | 7 |
| Doberman | Miniature pinscher | 7 |

**The dataset's contradictory label pairs survive pretraining.** Eskimo dog
against Siberian husky and English foxhound against Walker hound appear in the
from-scratch model's top confusions, in Swin-T's weakest classes, and here. All
three are pairs the duplicate analysis showed the dataset itself labels two
different ways.

This is worth stating directly in the discussion: **transfer learning raises the
ceiling but does not repair the data.** No amount of pretraining teaches a model
a distinction the labels do not make consistently.

The miniature poodle and Lhasa against Shih-Tzu failures are the other familiar
category, classes separated by size or by fine coat detail that a 224 by 224
square crop cannot represent.

## 5. The simplified split

Merging to 114 classes raises accuracy by **1.21 points**, from 67.53 to 68.74.
The same pattern appears across all three pretrained models:

| Model | Unclean (120) | Simplified (114) | Gain |
|---|---|---|---|
| MobileNetV3-Large | 67.53 % | 68.74 % | +1.21 |
| EfficientNetV2-S | 83.12 % | 84.54 % | +1.42 |
| Swin-T | 88.03 % | 89.92 % | +1.89 |

Consistent, modest, and in every case explainable by the merge removing the
hardest decisions rather than by any model improving.

## 6. Deployment note

Inference takes **0.57 ms per image** on the RTX 4050, with 4.36 million
parameters. That supports the deployment framing: this model reaches 67.5 per
cent on a 120 way fine-grained task at a size and speed suitable for ordinary
hardware, where Swin-T needs six times the parameters for a further 20 points.

## 7. Where the files are

`Lindani_Model/mobilenet/results/runs/mobilenet_unclean/` and
`.../mobilenet_simplified/`, each containing `run_config.json`, `metrics.csv`,
`history.json`, `training_curves.png` and an `eval_test/` folder with the
metrics, per-class results, confusion matrix and sample predictions.
