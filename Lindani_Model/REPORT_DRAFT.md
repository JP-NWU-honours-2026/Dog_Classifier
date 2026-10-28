# Report draft: Lindani's sections

Drafted 27 October 2026. **This is a draft to rewrite, not to submit.** Rewrite it in your
own words, both because the marker will compare it against how you speak in the oral, and
because you have to defend every sentence.

Every number here comes from a file in this repository. The mapping is in the last
section, so anything can be checked rather than trusted.

Placeholders needing results that do not exist yet are marked **[PENDING]**.

---

## 1. Dataset and preparation

### 1.1 Source and access

The dataset is Stanford Dogs (Khosla *et al.*, 2011), obtained from the Kaggle mirror
`jessicali9530/stanford-dogs-dataset`. It contains 20,580 photographs across 120 dog
breeds, with one PASCAL VOC bounding box annotation per photograph. All 20,580
photographs and all 20,580 annotation files were verified present locally, and none failed
to decode.

The dataset was assembled from ImageNet (Deng *et al.*, 2009) rather than collected
independently, and its breed folders are named by ImageNet WordNet synset identifiers, for
example `n02085620-Chihuahua`. This inheritance matters to the comparison and is returned
to in the discussion.

### 1.2 Licence and usage conditions

The dataset authors publish no licence. The project page provides download links, the
citations the authors request and baseline results, but no terms of use, copyright
statement or licence of any kind. The Kaggle mirror records its licence as "Other
(specified in description)", and that description restates provenance and citation rather
than terms.

The operative terms are therefore ImageNet's, since the images and annotations are taken
from ImageNet. The ImageNet terms of access state that "Researcher shall use the Database
only for non-commercial research and educational purposes", permit access to be shared
only with colleagues who accept the same terms, and transfer no copyright in the
photographs, which remain the property of their original owners.

Use in this project is therefore within the terms, as a university module is
non-commercial research and education. No model weights or derived artefacts from this
work may be used commercially.

### 1.3 Ethics

The photographs were collected from the web and many contain identifiable people, private
property and vehicles. No consent was obtained from those people, either by the dataset
authors, so far as either source records, or by us. Our own figures show this plainly: the
photographs in which the dog occupies least of the frame are typically dominated by
handlers, spectators or vehicles.

**[PENDING]** The ethics paragraph for the group's own photograph set, covering owner
consent, bystanders and POPIA, is Member X's section and should be kept separate from this
one.

### 1.4 Contents and class balance

| Measure | Value |
|---|---|
| Photographs | 20,580 |
| Breeds | 120 |
| Photographs per breed | 148 (redbone) to 252 (Maltese dog) |
| Mean, median per breed | 171.5, 159.5 |
| Imbalance ratio | 1.70 to 1 |
| Colour modes | 20,579 RGB, 1 RGBA |

The distribution is skewed rather than uniform: 45 of the 120 breeds fall in the lowest
bin, which is why the median of 159.5 sits well below the mean of 171.5. In practical
terms the dataset supplies roughly 150 photographs per breed as a floor, with a minority of
breeds supplied more generously.

An imbalance of 1.70 to 1 is mild. Class weighting was therefore not applied, and the
decision was made before training rather than after seeing results. Macro averaged F1 is
reported throughout alongside accuracy so that performance on the smaller breeds cannot be
hidden by performance on the larger ones.

The single RGBA photograph was identified during the audit and is converted to RGB at load
time. Left unhandled it would have failed partway through a training run.

### 1.5 Image properties, and what they imply for preprocessing

Widths range from 97 to 3,264 pixels and heights from 100 to 2,562, with medians of 500 and
375 respectively. The distribution shows a hard boundary at 500 pixels on both axes, which
indicates that the photographs were downscaled so that the longest side is at most 500
pixels before publication. The fine detail on which breed identification depends, such as
coat texture and ear outline, was therefore already reduced before the data reached us.

A further 1,577 photographs (7.66 per cent) are smaller than the 224 pixel model input on
at least one side, and are enlarged at load time, which softens them.

68.72 per cent of photographs are landscape, 25.95 per cent portrait and 5.33 per cent near
square, with a median shape of 1.33 wide to 1 tall. The shared preprocessing resizes
directly to 224 by 224 rather than cropping, so non-square photographs are distorted rather
than trimmed. This was adopted unchanged for comparability across the three models, and is
recorded here as a known distortion rather than presented as ideal.

### 1.6 How the labels were obtained and checked

The labels are inherited from ImageNet's categories and were not relabelled by the dataset
authors. Neither the project page nor the Kaggle mirror reports any verification pass.

Two checks were carried out.

First, the near-duplicate analysis (section 1.8) identified **50 pairs of confirmed
duplicate photographs that carry two different breed labels, across 40 of the 120 breeds**.
Since the two members of such a pair are the same photograph, at least one label in each
pair is wrong. The affected pairs are not random: they are precisely the breeds that are
hardest to separate, including Eskimo dog with Siberian husky, Walker hound with English
foxhound, and miniature with standard schnauzer. A model is therefore scored against
distinctions the dataset itself does not apply consistently, and perfect accuracy on this
dataset is not attainable.

Second, a random sample was drawn with seed 42 and inspected by eye against its label, each
photograph marked agree, disagree or unsure. **20 of the planned 100 photographs have been
judged so far: 19 agree, 1 disagree, 0 unsure.** That is an observed disagreement rate of
**5.0 per cent, with a 95 per cent Wilson interval of 0.9 to 23.6 per cent.**

The interval is very wide because the sample is small, and the honest reading is that the
audit so far is consistent with anything from a very low error rate to roughly one label in
four being wrong. It should be reported as a partial audit with its sample size stated,
not as a precise figure. **[PENDING: complete the remaining 80 to narrow the interval.]**

The disagreement found was photograph 11, labelled collie, from the training partition.

Two points of method. The auditor is not a breed expert, which is why "unsure" is a
permitted verdict and is reported rather than hidden. And photographs with no verdict are
excluded from the denominator rather than counted as correct, since an unexamined
photograph is not evidence of a correct label.

### 1.7 The split

The 70/15/15 split was generated by a group member and adopted unchanged by all three
models so that the comparison is of models rather than of data. It was verified
independently rather than assumed, and five checks were recorded:

1. All 20,580 photographs named in the split exist.
2. Rerunning the splitting procedure with seed 42 reproduces the split exactly, photograph
   for photograph and in the same order.
3. No photograph appears in more than one partition, and none is listed twice.
4. Every photograph's label agrees with its breed folder, and the class index list matches
   the sorted folder list.
5. Every one of the 120 breeds appears in all three partitions.

The resulting partitions are 14,355 training (69.8 per cent), 3,025 validation (14.7 per
cent) and 3,200 test (15.5 per cent) photographs. The split is stratified by breed, with
per-breed training shares between 69.4 and 70.0 per cent. The small departure from exactly
70/15/15 arises because the per-breed counts are rounded down for training and validation
and the remainder is assigned to test.

The audited file is identified by its SHA-256 digest
`4d9109a7a3f65ef2acf32feeced6893ab206b87718b0065cad97c1c9b9e9fd97`, so the report refers to
a specific file rather than to a procedure.

### 1.8 Near-duplicate photographs and leakage

Section 7.1 of the specification requires that near-identical images are not divided across
training and test. The split as supplied did not address this, so it was measured.

Every photograph was reduced to two independent 64 bit perceptual fingerprints, a
difference hash and a perceptual hash, and additionally a SHA-256 digest of the file bytes.
All 20,580 by 20,580 comparisons were carried out.

**Exact copies, which require no judgement:** 89 groups of byte-identical files, and 37 of
those pairs fall in different partitions.

**Near duplicates:** the threshold for "near identical" is a judgement that must be
defended, so seven thresholds were evaluated and a second, independent hashing method was
used as a cross check. Agreement between the two methods falls from 93.9 per cent at
distance 0 to 41.4 per cent at distance 8 and 13.4 per cent at distance 10, and inspection
of sampled pairs confirms that pairs flagged beyond distance 6 are frequently different
photographs. The adopted rule therefore requires **both** methods to agree: difference hash
distance of 5 or less **and** perceptual hash distance of 10 or less.

Under that rule there are 173 confirmed duplicate pairs, 74 of them split across
partitions. **34 test photographs (1.06 per cent) and 35 validation photographs (1.16 per
cent) have a confirmed duplicate inside the training partition.**

The leakage is therefore real but small. Two options were available: regenerate the split,
which would have required all three members to retrain, or retain the shared split and
additionally report a clean test score with the affected photographs excluded. The second
was adopted, because it preserves comparability across the three models, needs no
coordination, and converts the problem into a measurement. Both the full test score of
3,200 photographs and the clean test score of 3,166 are reported.

The conservative figure is 25 test photographs, being those at distance 0 where every pair
is byte identical or visually indistinguishable; the inclusive figure under the adopted
two-method rule is 34. Both are reported rather than a single number, since the difference
between them is the judgement the threshold represents.

---

## 2. Methodology: the baseline convolutional network

### 2.1 Role in the comparison

Three models from three families are compared: a small convolutional network trained from
scratch, EfficientNetV2-S with ImageNet pretrained weights, and Swin-T with ImageNet
pretrained weights. The from-scratch network is the control. It is the only model that has
never seen a photograph outside the training partition, so the difference between it and
the other two is the measurement of what transfer learning contributes on this task.

That role imposes two constraints which pull against each other. The network must be a fair
attempt, since a deliberately weak control would inflate the apparent benefit of
pretraining. It must also remain a small network trained from scratch, since turning it
into a deep residual network would blur the distinction it exists to draw.

The governing constraint on the design is that 14,355 training photographs across 120
breeds is approximately 120 photographs per breed, which is very little for a network
learning visual features from random initialisation.

### 2.2 Architecture

Input is 224 by 224 pixels in three colour channels, matching the other two models.

| Block | Layers | Channels | Output | Parameters |
|---|---|---|---|---|
| 1 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 3 to 32 | 112 x 112 | 928 |
| 2 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 32 to 64 | 56 x 56 | 18,560 |
| 3 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 64 to 128 | 28 x 28 | 73,984 |
| 4 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 128 to 256 | 14 x 14 | 295,424 |
| 5 | 3x3 conv, batch norm, ReLU, 2x2 max pool | 256 to 512 | 7 x 7 | 1,180,672 |
| Head | global average pool, dropout 0.3, linear | 512 to 120 | 120 | 61,560 |

**Total: 1,631,128 trainable parameters**, counted from the constructed model rather than
estimated. For comparison, EfficientNetV2-S and Swin-T are of the order of 21 and 28
million parameters respectively **[PENDING: confirm both from the library]**, so the
control is roughly fifteen to twenty times smaller than the models it is compared against.

Design decisions, each with the alternative considered:

- **Five blocks with widths doubling from 32 to 512.** Each block halves the spatial
  dimensions, so five blocks take 224 down to 7, which is the range required to progress
  from edges to whole-animal shapes. Three blocks, as initially suggested, would have
  stopped at 28 by 28 after three convolutions and would have behaved largely as a texture
  detector. Doubling the convolutions per block throughout was rejected at design time
  because it raises the parameter count to roughly 4.6 million against approximately 120
  photographs per class; it is instead tested as a controlled variant in iteration 2.
- **3 by 3 kernels throughout.** Two stacked 3 by 3 layers cover the same receptive field
  as one 5 by 5 layer with fewer parameters and an additional non-linearity. A 7 by 7
  stride 2 opening layer, as used by ResNet, was rejected because it discards fine detail in
  the first layer, and fine detail is the task.
- **Batch normalisation after every convolution**, without which a network trained from
  scratch converges slowly and is sensitive to learning rate. Group normalisation was
  unnecessary at batch size 64.
- **2 by 2 max pooling**, which preserves the strongest local response. Average pooling
  dilutes it, and strided convolutions add parameters.
- **Global average pooling head.** The alternative, flattening to a fully connected layer,
  adds approximately 524,000 parameters in a single layer, roughly nine times the entire
  head adopted, and fully connected layers on small datasets are where overfitting begins.
  The cost of global average pooling is that spatial position is discarded, which is
  acceptable for breed identification.
- **He initialisation**, appropriate for ReLU networks, which matters more here than for the
  pretrained models because the starting weights are chosen rather than inherited.

### 2.3 Preprocessing and augmentation, held identical across models

Training photographs receive a random horizontal flip, a rotation of up to 15 degrees and
mild colour jitter on brightness, contrast and saturation. Validation and test photographs
receive resizing and normalisation only, since evaluating on randomly altered photographs
would add noise to the measurement.

Vertical flips and large rotations were excluded because dogs are photographed upright, so
such transformations are not realistic for the domain.

All photographs are normalised using the ImageNet channel means and standard deviations.
This is applied to the from-scratch model as well, despite it never having seen ImageNet,
so that all three models receive numerically identical input.

---

## 3. Experimental setup and reproducibility

### 3.1 Configuration

| Setting | Value |
|---|---|
| Framework | PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 |
| Python | 3.13.15 |
| Hardware | NVIDIA Tesla T4, 14.6 GB, CUDA 12.8 (Kaggle) |
| Input size | 224 x 224 |
| Batch size | 64 |
| Optimiser | AdamW |
| Learning rate | 1e-3 |
| Schedule | Cosine annealing over the 60 epoch budget |
| Weight decay | 1e-4 |
| Dropout | 0.3 |
| Loss | Cross entropy, label smoothing 0.0 |
| Epoch budget | 60 |
| Early stopping | Validation macro F1, patience 5 |
| Mixed precision | Enabled |
| Random seed | 42 |
| Training time | 64.7 minutes |

Every value above is written automatically to `run_config.json` by the training script at
run time, together with the library versions and the GPU reported by the session, rather
than being reconstructed afterwards.

The learning rate and epoch budget differ from those used for the two pretrained models.
This is expected: pretrained models begin with useful features and converge at lower
learning rates in fewer epochs, whereas a network initialised randomly requires
substantially longer. The specification permits such differences where they are stated and
where no model is unfairly advantaged. Wall clock training time is reported for each model
so that the comparison remains visible rather than asserted.

### 3.2 Determinism

Seeding fixes the data order, the initialisation and the augmentation, so a run is
repeatable in structure. It does not make GPU arithmetic bitwise identical between runs,
because some CUDA operations accumulate in a non-fixed order. Deterministic algorithms can
be forced by a command line flag at a cost in speed, and were not used for the reported
runs.

The size of that variation was measured rather than assumed. Both iteration 2
configurations were, by accident, executed twice with identical code and identical seed,
once in an interactive session and once in a committed run:

| Configuration | Execution 1 | Execution 2 | Spread |
|---|---|---|---|
| 2A, regularised | 0.4944 | 0.4911 | 0.3 points |
| 2B, deeper | 0.5527 | 0.5649 | 1.2 points |

Early stopping also halted at different epochs, 60 against 57 and 51 against 59, since a
small difference in trajectory changes where the patience window expires.

**A difference of about one point in validation macro F1 is therefore not evidence of
anything in this setup.** All comparisons in section 4 are read against that floor: the
6.8 point gain of iteration 2B is far outside it, and the 0.6 point loss of iteration 2A is
well inside it. The figures reported throughout are those from the committed runs, which
are reproducible from a saved notebook version, rather than from the interactive session.

### 3.3 Protocol for iteration

The test partition was not read during model development. All comparisons between
iterations were made on the validation partition, and the test partition was scored once,
after the final model had been selected. The evaluation script enforces this: it evaluates
validation by default and reads test only when explicitly instructed, and refuses to score
test twice without an override.

This matters because repeatedly scoring on test and then adjusting the model converts the
final figure into a measure of how well the model was tuned against the test photographs,
which is not what the report would be claiming.

---

## 4. Results

### 4.1 Iteration 1, validation

| Metric | Value |
|---|---|
| Accuracy | 50.51 per cent (95 per cent interval 48.73 to 52.29) |
| Top-5 accuracy | 81.36 per cent |
| Macro F1 | 0.4969 |
| Macro precision | 0.5043 |
| Macro recall | 0.4998 |
| Weighted F1 | 0.5012 |
| Best epoch | 54 of a 60 epoch budget |
| Stopping | Early stopping at epoch 59; the cap did not bind |

Chance performance on 120 classes is 0.83 per cent, so the network performs approximately
sixty times better than chance from random initialisation.

Training accuracy reached 67.5 per cent against validation accuracy of 50.3 per cent, and
training loss fell to 1.16 against validation loss of 1.90. Validation loss is
approximately flat from epoch 45 onward while training loss continues to fall, which
identifies overfitting rather than insufficient capacity as the binding constraint.

That the epoch cap did not bind is reported explicitly, since a cap that silently binds is
an unnoticed handicap on a model.

### 4.2 Iteration 2

Two variants were trained, each changing one factor, each judged on validation only.

- **Run A, stronger regularisation:** dropout raised from 0.3 to 0.5 and weight decay from
  1e-4 to 1e-3, architecture unchanged. Directed at the overfitting identified in 4.1.
- **Run B, greater capacity for fine discrimination:** a second convolution in blocks 4 and
  5, raising the parameter count from 1,631,128 to 4,581,784. Directed at the gap between
  top-5 and top-1 accuracy.

| Run | Parameters | Dropout | Weight decay | Accuracy | Top-5 | Macro F1 | Train minus validation accuracy |
|---|---|---|---|---|---|---|---|
| 1, baseline | 1,631,128 | 0.3 | 1e-4 | 50.51 % | 81.36 % | 0.4969 | 15.8 points |
| 2A, regularised | 1,631,128 | 0.5 | 1e-3 | 50.05 % | 81.62 % | 0.4911 | **8.9 points** |
| 2B, deeper | 4,581,784 | 0.3 | 1e-4 | **56.96 %** | **84.89 %** | **0.5648** | **33.0 points** |

**Run A behaved exactly as intended and achieved nothing.** Stronger regularisation nearly
halved the overfitting gap, from 15.8 points to 8.9, and accuracy did not move: 50.05 per
cent against 50.51, a difference inside run-to-run variation (section 3.2). The treatment
worked on the symptom and left the result unchanged.

**Run B did the opposite and won.** Tripling the parameters **doubled** the overfitting gap
to 33.0 points and raised accuracy by 6.5 points and macro F1 by 6.8.

The two runs together refute the natural reading of iteration 1's loss curves. A widening
gap between training and validation loss was taken to mean that overfitting was the binding
constraint; the experiment shows it was not. The size of the overfitting gap turned out to
be a poor guide to what would improve the model, and capacity for fine discrimination was
the real limit. **This is the central methodological finding of the iteration sequence, and
it was only visible because the regularisation hypothesis was tested rather than assumed.**

Run 2B was selected on validation macro F1.

### 4.3 Iteration 3

Two further variants, again one factor each, again validation only.

| Run | Parameters | Val macro F1 | Val accuracy | Train minus validation | Outcome |
|---|---|---|---|---|---|
| 3A, deeper and regularised | 4,581,784 | 0.5602 | 56.56 % | 27.2 points | No change |
| 3B, deeper, 120 epoch schedule | 4,581,784 | 0.5270 | 53.82 % | 23.9 points | Worse |

**3A** is 0.47 points below iteration 2B, inside the measured run-to-run floor, so it is not
a difference. The overfitting gap fell again, from 33.0 to 27.2 points, with no gain in
accuracy. Regularisation has now behaved identically on two architectures: it does what it
claims to the gap, and nothing to the result.

**3B** reached only 0.5270, and the reason is instructive. It stopped at epoch 47 with its
best epoch at 37, so it never reached the annealed phase of its schedule. Spreading the
cosine schedule over 120 epochs keeps the learning rate high for far longer, validation
plateaus while the steps are still large, and early stopping expires before the schedule
anneals. Part of iteration 2B's advantage therefore came from its cosine schedule
completing inside its epoch budget, which was not an intended variable.

Iteration 2B was selected as the final model. Iterations 2B and 3A are statistically
indistinguishable, and the simpler configuration was preferred rather than treating a 0.47
point difference as a result.

### 4.4 Test set, scored once

The test partition was read once, with iteration 2B's weights from epoch 54.

| Metric | Value |
|---|---|
| Accuracy | **53.84 per cent** (95 per cent interval 52.11 to 55.57) |
| Top-5 accuracy | 84.78 per cent |
| Macro F1 | 0.5320 |
| Macro precision | 0.5403 |
| Macro recall | 0.5341 |
| Weighted F1 | 0.5361 |

Chance is 0.83 per cent, so the network is approximately 65 times better than chance
having never seen a photograph outside the training partition.

Test accuracy is 3.1 points below the 56.96 per cent measured on validation. Some of that
is ordinary sampling variation between two partitions of about 3,000 photographs, and some
is the selection effect of choosing the best of five runs on validation. The validation
figure should not be quoted as the result for this reason.

**The leakage measurement from section 1.8:**

| Partition | Photographs | Accuracy |
|---|---|---|
| Full test | 3,200 | 53.84 per cent |
| Clean test, duplicates removed | 3,166 | 53.63 per cent |
| Difference | | **+0.21 percentage points** |

Removing the 34 test photographs with a confirmed duplicate in training costs 0.21
percentage points. The duplicate leakage inside the dataset is therefore real but
negligible for this model, which is itself worth reporting: it establishes that
dataset-internal duplication is **not** the mechanism that inflates published accuracies on
Stanford Dogs. The mechanism of concern is the overlap between the test photographs and the
ImageNet pretraining corpus, which affects the two pretrained models and not this one.

### 4.5 Comparison with the pretrained models [PENDING]

Insert the three-model table once the other members' figures are available: accuracy, macro
F1, parameter count, training time and inference cost.

---

## 5. Discussion and error analysis

### 5.1 The nature of the failure

On the test set, top-5 accuracy of 84.78 per cent against top-1 accuracy of 53.84 per cent
shows that the correct breed is among the model's five highest scoring classes in roughly
six cases out of seven, while being its first choice only about half the time.

Of the 3,200 test photographs, 1,477 are misclassified. Of those errors, only 114 (7.7 per
cent) are made with 90 per cent confidence or more, so the model is for the most part
uncertain when it is wrong rather than confidently wrong. The true breed falls outside the
top five entirely for 487 photographs, 15.2 per cent of the test set, and those are the
cases where the model has genuinely failed rather than merely hesitated between similar
breeds.
The network has therefore learned the general visual structure of dogs; what it fails at is
the final discrimination between similar breeds. The failure is one of discrimination
rather than of recognition, which is the expected shape of a fine-grained problem and is
what the remaining analysis addresses.

### 5.2 Errors attributable to the dataset rather than the model

Of the twenty most frequent confusions on the test set, **seven are breed pairs that the
dataset itself labels inconsistently**, as established in section 1.6:

| True breed | Predicted | Times |
|---|---|---|
| Eskimo dog | Siberian husky | 10 |
| Siberian husky | Eskimo dog | 10 |
| Toy poodle | Miniature poodle | 8 |
| Miniature schnauzer | Standard schnauzer | 7 |
| Standard schnauzer | Miniature schnauzer | 7 |
| Miniature poodle | Toy poodle | 6 |
| Walker hound | English foxhound | 5 |

Each of these pairs appears in the 50 confirmed duplicate pairs that carry two different
breed labels. The model is therefore penalised for failing to reproduce distinctions the
annotations themselves do not apply consistently, and five of the seven are symmetric,
meaning the confusion runs in both directions as would be expected when the boundary is
genuinely unclear rather than when the model simply prefers one class.

The confusion submatrix of the fifteen weakest breeds
(`eval_test/fig_confusion_worst.png`) shows these as paired blocks sitting off the
diagonal. For two breeds the confusion with their twin **exceeds** their own correct
count: Eskimo dog is classified as Siberian husky 10 times against 5 correct, and toy
poodle is classified as miniature poodle 8 times against 7 correct. In other words the
model's most likely answer for an Eskimo dog is "Siberian husky". Given that these two
breeds appear in the dataset's own contradictory label pairs, this is better described as
a disagreement about where the class boundary lies than as a failure of recognition.

The same figure shows a second family of errors that is not about labelling at all:
redbone with Rhodesian ridgeback, and American Staffordshire terrier with boxer and Great
Dane. These are short-coated, similarly proportioned dogs separated mainly by size and
head shape, which returns to the scale problem described below.

A second category is structural. The weakest class is standard poodle at F1 0.062, and its
predictions are distributed across miniature poodle and several other curly or
light-coloured breeds. Standard, miniature and toy poodle are separated by height rather
than by appearance, and a 224 by 224 crop carries no scale reference, so this distinction
is close to unlearnable from a single photograph by construction.

A third category concerns photographs containing more than one dog, which the audit
measured at 6.60 per cent of the dataset, with up to six dogs in a single photograph under
one label. Inspection of the most confident errors shows cases where the predicted breed is
visibly present in the photograph but is not the labelled animal. Such errors are counted
as model failures by any standard metric, though the model's output is arguably correct for
an animal in the frame.

### 5.3 Remaining errors

The remaining confusions are between breeds that are genuinely difficult to separate
visually, including Siberian husky with malamute, Italian greyhound with whippet, and
Scottish deerhound with Irish wolfhound. These are the errors a stronger model would be
expected to reduce, and they are the appropriate basis for comparison with the pretrained
models.

### 5.4 Interpreting the comparison with pretrained models **[PENDING once their figures exist]**

The claim to make is narrow and defensible. Stanford Dogs is constructed from ImageNet, so
models initialised with standard ImageNet pretrained weights have encountered the test
photographs, with labels, during pretraining. The test partition is therefore not held out
with respect to those models, and their reported accuracy measures something other than
generalisation to unseen photographs.

The claim to avoid is that transfer learning is illegitimate. It is standard practice and
usually advantageous. The precise statement is that for this particular dataset the test
partition is not independent of the pretraining corpus, which is why the from-scratch
control is informative and why the gap between the models should not be read as a pure
measure of architecture quality.

### 5.5 Limitations

1. The dataset's photographs were downscaled to at most 500 pixels on the longest side
   before publication, limiting the fine detail available for a fine-grained task.
2. Resizing to a square distorts the 94.67 per cent of photographs that are not square.
3. At least 50 pairs of photographs carry contradictory labels, so perfect accuracy is not
   attainable.
4. 1.06 per cent of the test partition has a confirmed duplicate in training; the clean test
   score quantifies the effect.
5. 6.60 per cent of photographs contain more than one dog under a single label.
6. Runs are seeded but not bitwise deterministic on GPU.
7. Single-seed results: differences of one to two points between iterations should not be
   over-interpreted **[note if the group runs multiple seeds]**.

---

## 6. References

Draft entries in NWU Harvard style. **Check each against the NWU referencing guide before
submission**, since conference paper formatting and the venue details both need confirming.

> DENG, J., DONG, W., SOCHER, R., LI, L.-J., LI, K. & FEI-FEI, L. 2009. ImageNet: a
> large-scale hierarchical image database. Paper presented at the IEEE Conference on
> Computer Vision and Pattern Recognition, Miami, FL.

> KHOSLA, A., JAYADEVAPRAKASH, N., YAO, B. & FEI-FEI, L. 2011. Novel dataset for
> fine-grained image categorization. Paper presented at the First Workshop on Fine-Grained
> Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition,
> Colorado Springs, CO.

Still to add, depending on what the final text cites: the VGG paper for stacked 3 by 3
convolutions, the batch normalisation paper, the AdamW paper, and the EfficientNetV2 and
Swin papers for the other two models.

---

## 7. Where every number comes from

| Claim | Source file |
|---|---|
| Counts, breeds, imbalance, dimensions, shapes, box statistics | `results/dataset_figures/dataset_summary.txt` |
| Per-breed counts before and after cleaning | `results/dataset_figures/table_class_counts.csv` |
| Split sizes, stratification, the five checks, SHA-256 | `results/split_check/split_check_summary.txt` |
| Exact copies, threshold sweep, the adopted rule, leakage counts | `results/duplicate_check/duplicate_check_summary.txt` |
| The 50 contradictory label pairs | `results/duplicate_check/label_conflicts.csv` |
| The 34 excluded test photographs | `results/duplicate_check/leaked_photos.csv` |
| Licence, terms, citations, ethics | `DATASET_PROVENANCE.md` |
| Architecture, parameter counts, design alternatives | `MODEL_DESIGN.md` |
| Every hyperparameter, library version and the GPU | `results/runs/baseline_cnn/run_config.json` |
| Per-epoch losses and accuracies | `results/runs/baseline_cnn/metrics.csv` |
| Validation metrics, best and worst breeds, top confusions | `results/runs/baseline_cnn/eval_val/metrics_summary.txt` |
| Per-breed precision, recall and F1 | `results/runs/baseline_cnn/eval_val/per_class_metrics.csv` |

### Figures available for the report

| Figure | File |
|---|---|
| One photograph from each of the 120 breeds | `results/dataset_figures/fig_class_examples.png` |
| Photographs where the dog fills least of the frame | `results/dataset_figures/fig_smallest_subjects.png` |
| Photographs per breed, and how breed sizes cluster | `results/dataset_figures/fig_class_distribution.png` |
| Dimensions and shapes against the 224 pixel input | `results/dataset_figures/fig_image_sizes.png` |
| Confirmed duplicates split across partitions | `results/duplicate_check/pairs_confirmed_crossing.png` |
| Evidence for the duplicate threshold | `results/duplicate_check/pairs_distance_*.png` |
| Training and validation curves, and the macro F1 curve | `results/runs/baseline_cnn/eval_val/fig_curves.png` |
| Confusion submatrix of the 15 weakest breeds | `results/runs/baseline_cnn/eval_val/fig_confusion_worst.png` |
| Correct predictions with confidence | `results/runs/baseline_cnn/eval_val/fig_correct.png` |
| Confident mistakes with what was predicted instead | `results/runs/baseline_cnn/eval_val/fig_wrong.png` |
