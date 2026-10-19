# Dataset source, licence and usage conditions

Notes for the dataset section of the report. Every claim below names where it came
from, so each one can be checked before it is written up. Compiled 12 September 2026.

## Source and access

| Item | Detail |
|---|---|
| Dataset | Stanford Dogs |
| Authors | Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, Li Fei-Fei (Stanford University) |
| Official page | http://vision.stanford.edu/aditya86/ImageNetDogs/ |
| Copy actually used | Kaggle mirror `jessicali9530/stanford-dogs-dataset`, uploaded by Jessica Li |
| Date accessed | Confirm and record the date the group downloaded it |
| Contents as published | 120 categories, 20,580 images, class labels and bounding boxes |
| Contents as verified locally | 120 folders, 20,580 images, 20,580 annotation files, 0 unreadable |

Note that the official page serves the images as a 757MB archive and the annotations
separately at 21MB, which matches what is in this repository.

## How the dataset was built

The official page states that the dataset "has been built using images and annotation
from ImageNet for the task of fine-grained image categorization". So both the
photographs and the bounding boxes are taken from ImageNet rather than collected fresh.
The Kaggle page repeats the same statement.

This is the fact the rest of the project rests on. The breed folder names are WordNet
synset identifiers, for example `n02085620-Chihuahua`, which are ImageNet's own class
identifiers.

## How the labels were obtained, and whether they were checked

The labels are inherited from ImageNet's own categories, not relabelled by the Stanford
authors. Neither the official page nor the Kaggle page reports any label verification
pass.

Our own checks found measured evidence of label error. `duplicate_check.py` identified
50 pairs of confirmed duplicate photographs carrying two different breed labels, across
40 of the 120 breeds. One label in each pair must be wrong. A 100 photograph sample
audit is still outstanding and will give an observed error rate.

## Licence and usage conditions

**The dataset authors publish no licence.** The official page has no licence, copyright
or terms of use section at all. It provides download links, the citations it asks for,
and baseline results. This absence is itself worth stating in the report rather than
inventing a licence for it.

**The Kaggle mirror lists its licence as "Other (specified in description)"**, and its
description specifies provenance and citations rather than any licence terms.

**The operative terms are therefore ImageNet's**, since the images and annotations come
from ImageNet. ImageNet's terms of access (https://www.image-net.org/download.php)
state, quoted directly:

> "Researcher shall use the Database only for non-commercial research and educational
> purposes."

The same terms also record that access may be granted to colleagues provided they
accept the terms, that Princeton and Stanford reserve the right to terminate access at
any time, that no warranties are given, and that a researcher employed by a for-profit
entity binds that employer to the terms. The terms do not transfer image copyright:
the photographs remain the property of their original copyright holders, which is
expected given they were collected from the web.

## What this means for this project

1. **Our use is within the terms.** A university module is non-commercial research and
   education.
2. **Nothing produced here may be used commercially**, including trained weights, since
   they derive from the data.
3. **Redistribution is the open question.** ImageNet's terms allow granting access to
   colleagues who accept the same terms. They do not obviously cover publishing the
   images in a public repository. This repository currently holds all 20,580
   photographs in `archive/`, and the repository is public on GitHub. Raise this with
   the group. It is a second, independent reason to remove the images from version
   control, alongside the repository size.
4. **The report must cite both papers**, primary and secondary, as the authors request.

## Ethics note for the report

The photographs were collected from the web and many contain identifiable people,
private property and vehicles, which the sample figures in `results/dataset_figures`
show plainly. No consent was obtained from those people by us or, as far as either page
records, by the dataset authors. This is worth one honest sentence in the ethics
paragraph, separately from the consent arrangements for the group's own photographs.

## Citations the authors ask for

Primary:

> Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset
> for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual
> Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition
> (CVPR), 2011.

Secondary:

> J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, ImageNet: A Large-Scale
> Hierarchical Image Database. IEEE Computer Vision and Pattern Recognition (CVPR),
> 2009.

Draft NWU Harvard versions, to be checked against the NWU referencing guide before use,
since the conference details and the guide's exact format for conference papers both
need confirming:

> KHOSLA, A., JAYADEVAPRAKASH, N., YAO, B. & FEI-FEI, L. 2011. Novel dataset for
> fine-grained image categorization. Paper presented at the First Workshop on
> Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern
> Recognition, Colorado Springs, CO.

> DENG, J., DONG, W., SOCHER, R., LI, L.-J., LI, K. & FEI-FEI, L. 2009. ImageNet: a
> large-scale hierarchical image database. Paper presented at the IEEE Conference on
> Computer Vision and Pattern Recognition, Miami, FL.

## Claims to verify before writing them up

These are leads, not settled facts. Check each one yourself.

1. That the official Stanford Dogs page really carries no licence statement, including
   inside the README linked from that page, which has not been read yet.
2. The exact current wording of the ImageNet terms of access, quoted from the page
   rather than from this file.
3. The date the group downloaded the data, and from which of the two sources.
4. The conference venue, city and dates for both papers, and the NWU Harvard format for
   a conference paper.
5. Whether ImageNet's terms permit the public redistribution now happening in
   `archive/`, which decides how firmly point 3 above is put to the group.
