"""
dataset.py

Feeds the shared split to the model.

Two things about this file carry marks, so they are worth stating plainly.

**Fairness.** The preprocessing and augmentation below are copied from
JP_Model/dataset.py so that all three models see identical photographs treated
identically. Resize to 224 by 224, ImageNet normalisation, and for training only
a horizontal flip, a rotation of up to 15 degrees and mild colour jitter. Do not
change these without the group agreeing, because a difference here would
invalidate the comparison. No vertical flip and no heavy rotation: dogs are
photographed upright, so those would be unrealistic for this domain.

**Portability.** The shared split file stores absolute paths from JP's computer,
so it cannot be read as it stands anywhere else. Only the breed folder and file
name are used, and the folder holding the photographs is discovered at run time.
That is what lets the same code run on this laptop and on a Kaggle notebook.

Optionally excludes the leaked test photographs found by duplicate_check.py, so
a clean test score can be reported next to the full one.

Run this file on its own for a self-test:

    python Lindani_Model/dataset.py

Needs: pip install torch torchvision
"""

import csv
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from split_check import IMAGES_DIR, LINDANI_DIR, MANIFEST_PATH, PILES, local_key

# Copied from JP_Model/dataset.py. Identical for all three models.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224

SEED = 42
LEAKED_PATH = LINDANI_DIR / "results" / "duplicate_check" / "leaked_photos.csv"

# Where the photographs might live. The first one that exists wins. The
# environment variable is checked first so a Kaggle notebook can point at a
# mounted dataset without editing any code.
CANDIDATE_IMAGE_DIRS = (
    os.environ.get("DOG_IMAGES_DIR"),
    IMAGES_DIR,                                                    # this repository
    "/kaggle/input/stanford-dogs-dataset/images/Images",            # Kaggle mirror
    "/kaggle/input/stanford-dogs-dataset/Images",
)


def find_images_dir():
    """
    Returns the folder holding the 120 breed folders, or raises with a message
    that says what to do about it.
    """
    for candidate in CANDIDATE_IMAGE_DIRS:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not find the breed folders. Set the DOG_IMAGES_DIR environment "
        "variable to the folder that contains folders like n02085620-Chihuahua. "
        f"Tried: {[str(c) for c in CANDIDATE_IMAGE_DIRS if c]}"
    )


def get_transforms():
    """
    Training augmentation and the plain validation or test preprocessing.

    Augmentation is applied to training photographs only. Validation and test
    photographs are resized and normalised, nothing more, because scoring a
    model on randomly altered photographs would add noise to the result.

    Normalisation subtracts the ImageNet channel means and divides by their
    standard deviations. It is applied here too, even though this model never saw
    ImageNet, so that all three models receive numerically identical input.
    """
    train = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_test = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train, val_test


def load_manifest():
    """
    Reads the shared split and returns, for each pile, a list of
    (breed_folder/file_name, label) pairs, plus the breed names in label order.

    The label numbers come from the manifest itself rather than being recounted
    here, so this model's class 7 is the same breed as JP's class 7.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = {
        pile: [(local_key(entry), entry["label"]) for entry in manifest["splits"][pile]]
        for pile in PILES
    }
    classes = manifest["classes"]
    names = [classes[str(i)]["display_name"] for i in range(len(classes))]
    return items, names


def leaked_keys(pile="test"):
    """
    The photographs in one pile that have a confirmed duplicate in training, as
    found by duplicate_check.py. Returns an empty set if that file is missing.
    """
    if not LEAKED_PATH.exists():
        return set()
    with open(LEAKED_PATH, newline="", encoding="utf-8") as f:
        return {row["photo"] for row in csv.DictReader(f) if row["pile"] == pile}


class DogBreedDataset(Dataset):
    """
    Hands one photograph at a time to the model: the image, its breed number and
    its short name. The name is carried through so that Stage 6 can show which
    photographs were misclassified.
    """

    def __init__(self, items, images_dir, transform):
        self.items = items
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, label = self.items[index]
        # convert("RGB") matters: one photograph in this dataset has a
        # transparency channel, and a four channel image would crash training.
        image = Image.open(self.images_dir / key).convert("RGB")
        return self.transform(image), label, key


def build_loaders(batch_size=64, num_workers=2, exclude_leaked=False, seed=SEED):
    """
    Returns train, validation and test loaders, the breed names, and a summary
    dictionary for the run log.

    exclude_leaked=True drops the test photographs that have a confirmed
    duplicate in training, giving the clean test set. The full test set stays
    available either way, so both scores can be reported.
    """
    images_dir = find_images_dir()
    items, names = load_manifest()
    train_tf, val_tf = get_transforms()

    dropped = 0
    if exclude_leaked:
        leaked = leaked_keys("test")
        before = len(items["test"])
        items["test"] = [pair for pair in items["test"] if pair[0] not in leaked]
        dropped = before - len(items["test"])

    datasets = {
        "train": DogBreedDataset(items["train"], images_dir, train_tf),
        "val": DogBreedDataset(items["val"], images_dir, val_tf),
        "test": DogBreedDataset(items["test"], images_dir, val_tf),
    }

    # A seeded generator, so the order photographs are shuffled into batches is
    # the same on every run.
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory,
                            generator=generator, drop_last=False),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory),
    }

    summary = {
        "images_dir": str(images_dir),
        "split_file": str(MANIFEST_PATH),
        "classes": len(names),
        "train_photographs": len(datasets["train"]),
        "val_photographs": len(datasets["val"]),
        "test_photographs": len(datasets["test"]),
        "leaked_test_photographs_excluded": dropped,
        "batch_size": batch_size,
        "input_size": INPUT_SIZE,
        "seed": seed,
    }
    return loaders, names, summary


def main():
    print("SELF-TEST\n")
    loaders, names, summary = build_loaders(batch_size=8, num_workers=0)
    for key, value in summary.items():
        print(f"  {key:<34} {value}")

    print(f"\n  first three breeds                 {names[:3]}")
    print(f"  last breed                         {names[-1]}")

    images, labels, keys = next(iter(loaders["train"]))
    print("\n  one training batch")
    print(f"    images                           {tuple(images.shape)}")
    print(f"    labels                           {tuple(labels.shape)}, "
          f"range {int(labels.min())} to {int(labels.max())}")
    print(f"    first photograph                 {keys[0]}")
    print(f"    pixel values                     {images.min():.2f} to {images.max():.2f}")

    # The label must agree with the breed folder the photograph came from, or
    # every score after this is meaningless.
    folder = keys[0].split("/")[0]
    claimed = names[int(labels[0])]
    print(f"\n  label check                        folder {folder} -> {claimed}")
    assert folder.split("-", 1)[1].replace("_", " ").title() == claimed, \
        "the label does not match the breed folder"

    clean = build_loaders(batch_size=8, num_workers=0, exclude_leaked=True)[2]
    print(f"\n  full test set                      {summary['test_photographs']}")
    print(f"  clean test set                     {clean['test_photographs']} "
          f"({clean['leaked_test_photographs_excluded']} leaked photographs excluded)")

    print("\nSelf-test passed.")


if __name__ == "__main__":
    main()
