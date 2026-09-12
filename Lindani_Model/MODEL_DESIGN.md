# Baseline CNN: design and justification

The architecture, decided before any model code was written, with the alternative
considered for each choice and what it would have cost. Agreed 12 September 2026.

## What this model is for

This is the **control** in a three way comparison. JP trains EfficientNetV2-S and
Sulaiman trains Swin-T, both starting from ImageNet pretrained weights. This network
starts from random numbers and has never seen a photograph outside the training split.

The difference between this model and the other two is the measurement of what transfer
learning is worth on this task. That places two requirements on the design, pulling in
opposite directions:

1. It must be a **fair attempt**. A deliberately weak network would inflate the apparent
   benefit of pretraining, and a marker would be right to say so.
2. It must stay a **small convolutional network trained from scratch**. Turning it into a
   residual network with dozens of layers would blur the comparison it exists to make.

## The constraint that shapes everything

Training data is 14,355 photographs across 120 breeds, which is about **120 photographs
per breed**. That is very little for a network learning vision from nothing. The other
two models arrive already knowing what fur, ears and legs look like.

So every decision below answers one question: how to get useful capacity without
memorising 120 photographs per class.

## The design

Input is 224 by 224 pixels, three colour channels, matching the other two models.

| Block | Layers | Channels | Output size | Parameters |
|---|---|---|---|---|
| 1 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 3 to 32 | 112 x 112 | about 0.9k |
| 2 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 32 to 64 | 56 x 56 | about 18.5k |
| 3 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 64 to 128 | 28 x 28 | about 73.9k |
| 4 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 128 to 256 | 14 x 14 | about 295.2k |
| 5 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 256 to 512 | 7 x 7 | about 1,180.2k |
| Head | global average pool, dropout 0.3, linear to 120 | 512 to 120 | 120 | about 61.6k |

**Total: roughly 1.63 million trainable parameters.** The exact figure will be printed
from the code and reported, rather than quoted from this table.

For context to confirm before use in the report: EfficientNetV2-S is roughly 21 million
parameters and Swin-T roughly 28 million. So this model is in the order of fifteen to
twenty times smaller than what it is being compared against, which is worth stating
plainly in the results discussion.

## The decisions, and what the alternatives would have cost

### 1. Five blocks, widths doubling from 32 to 512

Each block halves the image, so 224 becomes 112, 56, 28, 14 and finally 7. Early blocks
see edges, middle blocks see fur texture and ear outlines, late blocks see whole head and
body shapes. Breed differences live at both ends, so the ladder has to reach the top.

- **Three blocks (32, 64, 128), as the implementation guide suggests.** Rejected. It
  stops at 28 by 28 after only three convolutions, so the network never builds up to
  whole dog shapes and works mostly as a texture detector. Expected cost: a weaker
  control, probably 10 to 20 per cent, which would overstate the benefit of pretraining.
- **Four blocks.** A defensible middle ground, and the fallback if training time becomes
  a problem.
- **Doubling the convolutions per block, VGG style.** Rejected. Two convolutions in each
  of the last three blocks would raise the parameter count to roughly 4.7 million, which
  on 120 photographs per class trades a little extra capacity for a lot more overfitting.

Widths double as the image halves, which is the standard convention: as spatial detail
is discarded, more pattern types are needed to describe what remains.

### 2. Kernel size 3 by 3 throughout

Two stacked 3 by 3 layers cover the same area as one 5 by 5 layer using fewer parameters
and adding an extra non-linearity, which is the argument made in the VGG paper (to be
cited properly in the report).

- **A 7 by 7 stride 2 opening layer, as ResNet uses.** Rejected. It cuts computation
  early by discarding fine detail in the very first layer, and fine detail is the whole
  task here. It would have saved training time we do not need to save.

### 3. Batch normalisation after every convolution

It rescales each layer's output so the next layer receives numbers in a sensible range.
Without it, a network trained from scratch converges slowly and is sensitive to the
learning rate.

- **Group normalisation.** Rejected as unnecessary. It earns its place when batches are
  too small for batch statistics to be stable, and the batch here is 64.
- **No normalisation.** Rejected. It would make the 40 to 60 epoch budget unreliable.

### 4. 2 by 2 max pooling

Max pooling keeps the strongest response in each patch, which suits the question "is
there a fur texture like this anywhere here".

- **Average pooling.** Rejected for the body of the network, since averaging dilutes a
  strong local response that matters.
- **Stride 2 convolutions that learn their own downsampling.** Rejected. It adds
  parameters and makes the "small and simple" claim harder to defend.

### 5. Global average pooling in the head

Each of the 512 final channels is averaged to a single number, giving a 512 number
summary, then dropout, then one linear layer to 120 outputs.

- **The implementation guide's head**: pool to 4 by 4, flatten to 2,048, then 2,048 to
  256, then 256 to 120. Rejected. That single fully connected layer is about 524,000
  parameters on its own, roughly nine times the whole head chosen here, and fully
  connected layers on small datasets are where overfitting starts.

The honest cost of global average pooling is that averaging discards **where** in the
frame each feature appeared. For breed identification that is an acceptable loss, since
a muzzle is a muzzle wherever it sits in the frame.

### 6. Regularisation

- **Dropout 0.3** before the final layer, so the model cannot lean on any single
  feature. The guide suggests 0.4; with global average pooling there is less to overfit,
  so 0.3 is the choice, and this is a value worth sweeping if time allows.
- **Weight decay 1e-4**, which discourages large weights.
- **Augmentation is not ours to choose.** The shared pipeline applies horizontal flip,
  rotation up to 15 degrees and mild colour jitter to training data only. It stays
  exactly as it is, because all three models must see the same photographs treated the
  same way. Vertical flips and heavy rotation are absent by design: dogs are
  photographed upright, so those transformations would be unrealistic for the domain.

### 7. Training configuration

| Setting | Value | Reason |
|---|---|---|
| Optimiser | AdamW | Standard, and separates weight decay from the gradient update |
| Learning rate | 1e-3 | Suits a network trained from scratch; the pretrained models use far lower rates, which is expected and must be stated |
| Schedule | Cosine decay | Large steps early, fine steps late |
| Batch size | 64 | Fits comfortably in Kaggle GPU memory and keeps batch norm statistics stable |
| Epochs | 40 to 60 | See below |
| Early stopping | On validation macro F1, patience about 5 | Stops when it stops improving, and macro F1 treats all 120 breeds equally |
| Loss | Cross entropy | Standard for single label classification. Whether label smoothing of 0.1 is used must match the other two models; group decision outstanding |
| Mixed precision | Yes | Faster on the Kaggle GPU at no accuracy cost |
| Seed | 42 | Same as the shared split |

**On epochs.** The implementation guide suggests 10 to 15. That is too few here.
Pretrained models converge in that range because they begin already knowing general
visual features. A network starting from random numbers needs far longer, so the budget
is 40 to 60 epochs with early stopping deciding the actual figure. The report must state
whether the cap was reached or early stopping triggered, since a cap that binds is an
unnoticed handicap.

## What is held identical across all three models

Differences here would make the comparison unfair, and fairness carries marks.

- The shared split, seed 42, verified by `split_check.py`
- Input size 224 by 224
- The augmentation pipeline and the ImageNet normalisation statistics
- The evaluation metrics, and the leaked photograph exclusion list if the group adopts it

What is **meant** to differ is the architecture, the learning rate and the epoch budget,
because those follow from the architecture. Each difference is stated with its reason, so
no model is quietly advantaged.

## Expected outcome, and how to read it

Roughly **20 to 30 per cent** top one accuracy on the 120 way test set, against a chance
level of 0.83 per cent. The pretrained models should land far higher.

That gap is the finding, but it must be stated carefully. It is not evidence that
transfer learning is cheating. It is evidence that starting from a model which has
already seen millions of photographs, **including these ones**, is worth a great deal on
a fine-grained task with only 120 examples per class. The contamination measured in
Stage 2 is part of the reason the gap is as large as it is, which is exactly why this
control matters.

## Questions to expect in the oral

- **"Why five blocks?"** The image halves at each block, so five blocks take 224 down to
  7. That is the range needed to get from edges to whole body shapes. Three would have
  stopped at texture.
- **"Why not deeper?"** 120 training photographs per class. More capacity would have been
  spent memorising rather than generalising.
- **"Why global average pooling?"** Parameter economy on a small dataset. The fully
  connected alternative costs roughly nine times more parameters in the head alone.
- **"Why a different learning rate and epoch count from the other two models?"** They
  start pretrained and converge quickly at low learning rates. This one starts from
  random numbers. The specification permits it where the reasoning is stated.
- **"How do you know the comparison is fair?"** Same split, same input size, same
  augmentation, same seed, same metrics. The split was independently verified rather than
  assumed.

## Still open

1. Label smoothing: use it or not, and the same choice for all three models.
2. Whether the group adopts the leaked photograph exclusion list, so a clean test score
   can be reported alongside the full one.
3. The parameter counts quoted for EfficientNetV2-S and Swin-T need confirming from the
   library rather than from memory.
