# Report sections: Swin-T (Sulaiman)

Draft text for the group report, written to drop into the recommended
structure from the brief (sections 4 to 7, plus references). Numbers come
from `results/runs/swin_t/`. Figure names refer to files in
`results/runs/swin_t/eval_test/`.

Items marked **[CHECK]** need confirming before submission. Items marked
**[GROUP]** depend on JP's or Lindani's final numbers.

---

## 4. Methodology: Swin Transformer (model family 3)

### 4.x.1 Why a transformer

The third model gives the comparison an architecture that does not rely on
convolution. A convolutional network sees an image through small filters
that slide over it, so each layer only looks at a local neighbourhood. A
transformer instead lets every part of the image compare itself with other
parts through self-attention, so relationships between distant regions,
such as ear shape against tail shape, can be learned directly.

The original Vision Transformer (ViT) (Dosovitskiy *et al.*, 2021) applies
attention between every pair of image patches, which makes the cost grow
with the square of the number of patches, and it needs very large datasets
to train well. The Swin Transformer (Liu *et al.*, 2021) was designed to fix
both problems, which makes it a better fit for a dataset of about 20,000
photographs.

### 4.x.2 How Swin works

1. **Patches.** The 224 x 224 image is cut into 4 x 4 pixel patches, giving
   a 56 x 56 grid. Each patch is turned into a vector of 96 numbers.
2. **Window attention.** Instead of every patch attending to every other
   patch, the grid is divided into non-overlapping windows of 7 x 7 patches,
   and attention is computed only inside each window. This makes the cost
   grow linearly with image size rather than quadratically.
3. **Shifted windows.** In every second block the window grid is shifted by
   half a window (3 patches). Patches that sat on the edge of a window are
   now in the middle of one, so information flows between neighbouring
   windows. Alternating normal and shifted windows lets the model build up
   a view of the whole image over several blocks. This is the "Shifted
   window" in the name.
4. **Hierarchy.** Between stages, neighbouring 2 x 2 patches are merged,
   halving the grid (56, 28, 14, 7) and doubling the vector size (96, 192,
   384, 768). This gives the same coarse-to-fine pyramid a CNN has: early
   stages see fur texture and edges, late stages see whole heads and bodies.
5. **Classification.** The final 7 x 7 grid of 768-number vectors is
   averaged into one 768-number summary of the photograph, which a linear
   layer maps to one score per class.

Swin-T ("Tiny") is the smallest standard variant: four stages with 2, 2, 6
and 2 blocks, and 3, 6, 12 and 24 attention heads. **[CHECK against Liu
*et al.*, 2021, Table 1]**

*(Suggested figure: a diagram of patch partition, the four stages with patch
merging between them, and one pair of regular and shifted windows. Draw it
ourselves or redraw from the paper with a citation.)*

### 4.x.3 Transfer learning approach

The model was loaded from torchvision with the ImageNet-1K weights
`Swin_T_Weights.IMAGENET1K_V1`, which reach about 81.5 per cent top-1
accuracy on ImageNet **[CHECK torchvision documentation]**. The original
1,000-class head was replaced by dropout (p = 0.3) followed by a new linear
layer from 768 features to 120 breeds. The new layer starts from random
weights; everything else keeps its pretrained weights.

The whole network was then fine-tuned, with no layers frozen. Freezing the
backbone would have been faster, but ImageNet's dog classes do not line up
exactly with the 120 breeds here, so letting the backbone adjust was
expected to help. To stop the large, noisy updates of the untrained head
from damaging the pretrained backbone early on, the learning rate was kept
small (1e-4) and ramped up from near zero over the first epoch.

| | Parameters |
|---|---|
| Pretrained backbone | 27,519,354 |
| New head (768 x 120 weights + 120 biases) | 92,280 |
| **Total, all trainable** | **27,611,634** |

For scale, this is about 17 times the size of the baseline CNN (1,631,128)
and of the same order as EfficientNetV2-S **[GROUP: JP's measured count]**.

---

## 5. Experimental setup: Swin-T

### 5.x.1 Shared with the other two models

All three models use the same split file (`split_manifest.json`, 70/15/15,
seed 42: 14,355 training, 3,025 validation and 3,200 test photographs), the
same 224 x 224 input with ImageNet normalisation, the same training-only
augmentation (horizontal flip, rotation up to 15 degrees, mild colour
jitter), the same loss (cross entropy, no label smoothing), the same metrics
and the same early-stopping rule on validation macro F1. The Swin-T scripts
read Lindani's leaked-photograph list directly, so the clean test set is the
same 3,166 photographs for every model.

### 5.x.2 Settings specific to Swin-T

| Setting | Value | Reason |
|---|---|---|
| Model | torchvision `swin_t`, `IMAGENET1K_V1` weights | Smallest Swin variant, similar size to EfficientNetV2-S |
| Optimiser | AdamW (Loshchilov & Hutter, 2019) | Standard for transformers; weight decay separate from the gradient step |
| Learning rate | 1e-4 | Small, so pretrained weights are adjusted rather than overwritten |
| Schedule | 1 epoch linear warm-up, then cosine decay to 0, per batch | Protects the backbone while the new head is random; fine steps at the end |
| Weight decay | 0.05 | Value used for Swin in the original paper **[CHECK]** |
| Batch size | 32, no gradient accumulation | Fits the 8 GB GPU with mixed precision |
| Epochs | Up to 15, early stopping patience 4 | Pretrained models converge quickly |
| Dropout | 0.3 before the new head | Same value as the baseline CNN |
| Gradient clipping | Max norm 1.0 | Keeps transformer updates stable |
| Mixed precision | Yes (float16 with gradient scaling) | Faster and lighter on the GPU |
| Seed | 42 | Same as the split |

The learning rate, schedule, weight decay, batch size and epoch budget differ
from the baseline CNN (1e-3, no warm-up, 1e-4, 64, up to 60 epochs). These
follow from the architecture: a network trained from random weights needs
larger steps and many more epochs than one that starts pretrained. The brief
allows this where it is justified, and neither model gets more data or a
different evaluation.

### 5.x.3 Software and hardware

Python 3.14, PyTorch 2.11.0 (CUDA 12.8), torchvision 0.26.0, NumPy 2.5.2,
Matplotlib 3.11.2, on Windows with an NVIDIA GeForce RTX 4060 Laptop GPU
(8 GB). Training took 19.0 minutes, about 71 seconds per epoch. Every
setting and library version is written automatically to `run_config.json`
at the start of each run.

GPU training is repeatable to within small differences, not bit for bit,
because some CUDA operations add numbers in a non-fixed order. A slower
fully deterministic mode is available (`train.py --deterministic`) but was
not used.

---

## 6. Results: Swin-T

### 6.x.1 Training behaviour

*(Figure: `fig_curves.png`, training and validation loss, accuracy, and
validation macro F1 per epoch.)*

Validation macro F1 reached 0.757 after one epoch, because the pretrained
backbone already recognised dogs and only the final breed decision had to
be learned. It rose to 0.839 by epoch 5 and 0.860 by epoch 10, then
flattened, gaining only 0.001 over the last four epochs. The best epoch was
the last (15), so the epoch cap was reached rather than early stopping. The
flat curve from epoch 12 indicates the model had converged, and a longer run
was not expected to change the result meaningfully.

Validation accuracy sits above training accuracy for the first four epochs.
This is expected: training accuracy is measured with augmentation and
dropout active and while the weights are still changing, whereas validation
is measured afterwards on unaltered photographs with dropout off.

From epoch 5 onwards training accuracy overtakes validation and ends at
97.7 per cent against 86.6 per cent. Validation loss was lowest at epoch 4
(0.513) and drifted up to 0.533 while validation accuracy kept improving.
The model was getting more photographs right but becoming more confident on
the ones it got wrong. This is mild overfitting; dropout, weight decay,
augmentation and early stopping kept it from hurting accuracy.

### 6.x.2 Test results

The test set was scored once, after training was finished.

| Metric | Validation (3,025) | Test (3,200) | Clean test (3,166) |
|---|---|---|---|
| Accuracy | 86.61 % (85.35 to 87.78) | **85.38 %** (84.11 to 86.56) | 85.50 % |
| Top-5 accuracy | 98.88 % | 98.56 % | |
| Macro precision | 0.8685 | 0.8542 | |
| Macro recall | 0.8649 | 0.8502 | |
| **Macro F1** | 0.8644 | **0.8490** | 0.8500 |
| Weighted F1 | 0.8650 | 0.8515 | |

*Brackets: 95 per cent Wilson confidence interval. Chance level for 120
classes is 0.83 per cent.*

Test accuracy is 1.2 points below validation, with overlapping confidence
intervals, so the model was not inadvertently tuned to the validation set.
Top-5 accuracy of 98.6 per cent shows that when the model is wrong, the
correct breed is almost always among its next four guesses.

### 6.x.3 Per-breed results and the confusion matrix

*(Figures: `fig_confusion_full.png`, the full 120 x 120 matrix; and
`fig_confusion_worst.png`, the 15 weakest breeds. Full per-breed table in
`per_class_metrics.csv`, suitable for an appendix.)*

The full confusion matrix has a strong diagonal, with the errors gathered
in a few small off-diagonal clusters rather than spread at random.

| Weakest breeds (test) | F1 | Strongest breeds (test) | F1 |
|---|---|---|---|
| Eskimo Dog | 0.410 | Chow | 1.000 |
| Collie | 0.439 | Pug | 0.984 |
| Miniature Poodle | 0.444 | Norwegian Elkhound | 0.983 |
| Lhasa | 0.528 | Saint Bernard | 0.981 |
| Siberian Husky | 0.557 | Old English Sheepdog | 0.980 |

| Most frequent test errors | Count |
|---|---|
| Eskimo Dog predicted as Siberian Husky | 11 |
| Miniature Poodle predicted as Toy Poodle | 9 |
| English Foxhound predicted as Walker Hound | 8 |
| Collie predicted as Border Collie | 8 |
| Lhasa predicted as Shih-Tzu | 7 |
| Border Collie predicted as Collie | 7 |
| American Staffordshire Terrier predicted as Staffordshire Bullterrier | 6 |
| Kuvasz predicted as Great Pyrenees | 6 |

---

## 7. Discussion and error analysis: Swin-T

### 7.x.1 The errors are look-alike breeds

Every one of the most frequent errors is between two breeds that people
also confuse. Several pairs differ mainly in size (Miniature and Toy Poodle;
Standard and Miniature Schnauzer), which a photograph without a reference
object cannot show. Others are close relatives with overlapping coats and
colours (Eskimo Dog and Siberian Husky; Collie and Border Collie; Lhasa and
Shih-Tzu). The strongest breeds, by contrast, have no close look-alike in the
dataset (Chow, Pug, Saint Bernard).

The baseline CNN struggles with several of the same pairs (Eskimo Dog and
Siberian Husky; Scottish Deerhound and Irish Wolfhound) **[GROUP: confirm
against Lindani's test results, and add JP's]**. Two very different
architectures failing on the same pairs suggests the limit lies in the data
rather than in either model.

### 7.x.2 Some "errors" are label errors

*(Figure: `fig_wrong.png`, the twelve most confident mistakes.)*

The most confident mistakes are often not mistakes. Among the photographs
the model got "wrong" with near 100 per cent confidence, the one labelled
French Bulldog shows a Boston Terrier, the one labelled Schipperke shows a
much larger dog that matches the Groenendael the model predicted, and the
one labelled Doberman looks like the Miniature Pinscher the model predicted.
**[CHECK: confirm each by eye before stating it]** The Stanford Dogs labels
come from ImageNet and were not independently verified (see the dataset
section), so a share of the measured error is label noise.

### 7.x.3 Leakage and memorisation

Lindani's duplicate check found 34 test photographs with a near-copy in the
training set. Removing them changes accuracy by only +0.13 points (85.38 to
85.50 per cent), which on its own suggests leakage did not matter. Looking at
those 34 photographs individually shows something more interesting:

| Leaked test photographs | Correct |
|---|---|
| Training copy has the **same** breed label (25) | 25 of 25 |
| Training copy has a **different** breed label (9) | 0 of 9 |

In all nine of the second group, the model predicted exactly the label of
the training copy. The model recalls photographs it was trained on: when the
labels agree this inflates the score, and when they disagree (a label error
in the dataset) it costs a mistake. The two effects happen to cancel in the
headline number.

A larger caveat cannot be measured here. Stanford Dogs was built from
ImageNet images (Khosla *et al.*, 2011), and Swin-T was pretrained on
ImageNet-1K, so the backbone has very likely seen many of the test
photographs before fine-tuning. Some of the gap between the pretrained models
and the baseline CNN therefore reflects prior exposure to these exact images,
not only better general visual knowledge. This applies equally to
EfficientNetV2-S.

### 7.x.4 Swin-T against the CNNs

| Model | Pretrained | Parameters | Test accuracy | Test macro F1 | Training time |
|---|---|---|---|---|---|
| Baseline CNN (Lindani) | No | 1.63 M | **[GROUP]** | **[GROUP]** | 64.7 min (Tesla T4) |
| EfficientNetV2-S (JP) | ImageNet-1K | **[GROUP]** | **[GROUP]** | **[GROUP]** | **[GROUP]** |
| Swin-T (Sulaiman) | ImageNet-1K | 27.6 M | 85.38 % | 0.8490 | 19.0 min (RTX 4060 Laptop) |

Points to make once the table is complete:

- **Transfer learning.** Swin-T reached 77 per cent validation accuracy
  after a single epoch, against the baseline's 50.5 per cent after 54. Most of
  that gap is the value of pretraining on a fine-grained task with only about
  120 training photographs per breed, with the ImageNet-overlap caveat above.
- **Attention against convolution.** EfficientNetV2-S and Swin-T are of
  similar size and both pretrained on ImageNet-1K, so the difference between
  them is the cleanest test of attention against convolution in this project.
  Report the difference with its confidence interval: if the intervals
  overlap, the honest conclusion is that the architectures perform alike on
  this task.
- **Recommendation.** Choose on test macro F1 first, then on size and speed.

### 7.x.5 Limitations

- One training run with one seed; no estimate of run-to-run variation.
- Hyperparameters were set from the literature and not tuned, to keep the
  test set untouched and the comparison fair.
- Photographs are squashed to 224 x 224 rather than cropped, which distorts
  body proportions, one of the cues that separates similar breeds.
- Label noise and ImageNet overlap, as above.

---

## References (NWU Harvard, draft) [CHECK format against the NWU guide]

DENG, J., DONG, W., SOCHER, R., LI, L.-J., LI, K. & FEI-FEI, L. 2009.
ImageNet: a large-scale hierarchical image database. Paper presented at the
IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL.

DOSOVITSKIY, A., BEYER, L., KOLESNIKOV, A., WEISSENBORN, D., ZHAI, X.,
UNTERTHINER, T., DEHGHANI, M., MINDERER, M., HEIGOLD, G., GELLY, S.,
USZKOREIT, J. & HOULSBY, N. 2021. An image is worth 16x16 words:
transformers for image recognition at scale. Paper presented at the
International Conference on Learning Representations (ICLR).

KHOSLA, A., JAYADEVAPRAKASH, N., YAO, B. & FEI-FEI, L. 2011. Novel dataset
for fine-grained image categorization. Paper presented at the First Workshop
on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and
Pattern Recognition, Colorado Springs, CO.

LIU, Z., LIN, Y., CAO, Y., HU, H., WEI, Y., ZHANG, Z., LIN, S. & GUO, B.
2021. Swin Transformer: hierarchical vision transformer using shifted
windows. Paper presented at the IEEE/CVF International Conference on
Computer Vision (ICCV), Montreal.

LOSHCHILOV, I. & HUTTER, F. 2019. Decoupled weight decay regularization.
Paper presented at the International Conference on Learning Representations
(ICLR), New Orleans, LA.

PASZKE, A. *et al.* 2019. PyTorch: an imperative style, high-performance deep
learning library. Paper presented at Advances in Neural Information
Processing Systems 32 (NeurIPS), Vancouver. [CHECK: full author list]

TORCHVISION. 2026. Models and pre-trained weights: swin_t.
https://pytorch.org/vision/stable/models/swin_transformer.html [CHECK URL
and date accessed]

---

## Oral presentation: Swin-T slide and likely questions

**One slide:** the Swin diagram, the three numbers (85.4 % test accuracy,
0.849 macro F1, 98.6 % top-5), and one sentence: "Pretrained attention model;
most errors are look-alike breeds, and some are label errors."

**Questions to expect:**

- *What does "shifted window" mean?* Attention is only computed inside 7 x 7
  patch windows to keep it cheap; every second block the windows move by half
  a window so neighbouring windows can share information.
- *Why not plain ViT?* ViT's attention cost grows with the square of the
  number of patches, and it needs far more data. Swin is linear and builds a
  hierarchy like a CNN.
- *Why a learning rate of 1e-4 and not 1e-3 like the CNN?* The backbone is
  already trained. Large steps would overwrite what it learned on ImageNet.
- *What is warm-up for?* The new head starts random, so its first gradients
  are large and noisy. Starting with a tiny learning rate stops that noise
  damaging the backbone.
- *The epoch cap was reached. Did you stop too early?* Validation F1 was flat
  from epoch 12 (0.863 to 0.864), so the model had converged.
- *Validation accuracy was higher than training at first. Is that a bug?* No.
  Training accuracy is measured with augmentation and dropout on, while the
  model is still learning.
- *Is the comparison fair?* Same split, seed, input size, augmentation, loss
  and metrics. Only settings that follow from the architecture differ, and
  each is justified.
- *Why might the score be optimistic?* ImageNet pretraining probably included
  these photographs, and 25 leaked test photos were all answered correctly.
