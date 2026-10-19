"""
dataset.py  (Sulaiman_Model, Swin Transformer)

Step 1: feeds the shared split to the model.

This is Lindani's Lindani_Model/dataset.py with two changes:
  1. It no longer imports from Lindani's split_check.py. The few constants it
     needed are defined here, so this folder stands on its own.
  2. The Kaggle folder guesses are gone, because training runs on a local
     RTX 4060. The DOG_IMAGES_DIR environment variable still works as an
     override if the photographs live somewhere else.

What must NOT change, because the three-way comparison depends on it:
  - the split file (split_manifest.json, seed 42, identical to JP's and Lindani's)
  - input size 224 x 224 and ImageNet normalisation
  - training-only augmentation: horizontal flip, rotation up to 15 degrees,
    mild colour jitter. Validation and test are only resized and normalised.

The leaked-photo list (test photos with a near-copy in training) is read from
Lindani's results folder, so the whole group uses one list, not three copies.

Self-test, from the repository root (the Dog_Classifier folder):

    python Sulaiman_Model/dataset.py
"""

import csv
import json
import os
from pathlib import Path, PureWindowsPath

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# Locations, all worked out relative to this file
# ---------------------------------------------------------------------------
SULAIMAN_DIR = Path(__file__).resolve().parent
REPO_ROOT = SULAIMAN_DIR.parent
IMAGES_DIR = REPO_ROOT / "archive" / "images" / "Images"
MANIFEST_PATH = SULAIMAN_DIR / "split_manifest.json"
LEAKED_PATH = REPO_ROOT / "Lindani_Model" / "results" / "duplicate_check" / "leaked_photos.csv"

# ---------------------------------------------------------------------------
# Shared settings. Identical for all three models.
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
SEED = 42
PILES = ("train", "val", "test")


def local_key(entry):
    """
    Turns one split-file entry into 'breed_folder/file_name'.

    The split file stores full paths from JP's laptop (C:\\Users\\jpret\\...),
    which don't exist on any other computer. Only the file name at the end is
    kept, and the photo is then looked up in this machine's images folder.
    """
    file_name = PureWindowsPath(entry["path"]).name
    return f"{entry['breed_folder']}/{file_name}"


def find_images_dir():
    """Returns the folder holding the 120 breed folders."""
    override = os.environ.get("DOG_IMAGES_DIR")
    for candidate in (override, IMAGES_DIR):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError(
        f"Could not find the breed folders at {IMAGES_DIR}. Either put the "
        "dataset there, or set DOG_IMAGES_DIR to the folder that contains "
        "folders like n02085620-Chihuahua."
    )


def get_transforms():
    """
    Training augmentation, and the plain preprocessing for validation and test.

    ImageNet normalisation matters more for this model than for Lindani's:
    Swin-T was pretrained on photos normalised exactly this way, so its
    weights expect numbers in this range.
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
    Reads the shared split. Returns, for each pile, a list of
    (breed_folder/file_name, label) pairs, plus the breed names in label order.

    Labels come straight from the split file, so class 7 here is the same
    breed as class 7 in JP's and Lindani's models.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = {
        pile: [(local_key(e), e["label"]) for e in manifest["splits"][pile]]
        for pile in PILES
    }
    classes = manifest["classes"]
    names = [classes[str(i)]["display_name"] for i in range(len(classes))]
    return items, names


def leaked_keys(pile="test"):
    """Photos in one pile with a confirmed copy in training (from Lindani's duplicate check)."""
    if not LEAKED_PATH.exists():
        return set()
    with open(LEAKED_PATH, newline="", encoding="utf-8") as f:
        return {row["photo"] for row in csv.DictReader(f) if row["pile"] == pile}


class DogBreedDataset(Dataset):
    """Hands the model one photo at a time: (image tensor, breed number, short name)."""

    def __init__(self, items, images_dir, transform):
        self.items = items
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, label = self.items[index]
        # One photo in the dataset has a transparency channel; convert("RGB")
        # stops a 4-channel image from crashing training.
        image = Image.open(self.images_dir / key).convert("RGB")
        return self.transform(image), label, key


def build_loaders(batch_size=32, num_workers=4, exclude_leaked=False, seed=SEED):
    """
    Returns (loaders, breed_names, summary).

    loaders is a dict with 'train', 'val' and 'test' DataLoaders.
    exclude_leaked=True drops the leaked test photos, giving the clean test set.
    summary is a dict that gets written into the run log for the report.
    """
    images_dir = find_images_dir()
    items, names = load_manifest()
    train_tf, val_tf = get_transforms()

    dropped = 0
    if exclude_leaked:
        leaked = leaked_keys("test")
        before = len(items["test"])
        items["test"] = [p for p in items["test"] if p[0] not in leaked]
        dropped = before - len(items["test"])

    datasets = {
        "train": DogBreedDataset(items["train"], images_dir, train_tf),
        "val": DogBreedDataset(items["val"], images_dir, val_tf),
        "test": DogBreedDataset(items["test"], images_dir, val_tf),
    }

    # Seeded generator: batches are shuffled in the same order on every run.
    generator = torch.Generator()
    generator.manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    # Keeps worker processes alive between epochs. Starting workers is slow on Windows.
    persistent = num_workers > 0

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=pin_memory,
                            generator=generator, persistent_workers=persistent),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=persistent),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory,
                           persistent_workers=persistent),
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
    print(f"  torch                              {torch.__version__}")
    print(f"  cuda available                     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu                                {props.name}, "
              f"{props.total_memory / 1024**3:.1f} GB")
    print()

    loaders, names, summary = build_loaders(batch_size=8, num_workers=0)
    for key, value in summary.items():
        print(f"  {key:<34} {value}")

    # Every photo listed in the split must exist on this machine.
    images_dir = Path(summary["images_dir"])
    items, _ = load_manifest()
    missing = [k for p in PILES for k, _ in items[p] if not (images_dir / k).is_file()]
    print(f"\n  photos missing on this machine     {len(missing)}")
    assert not missing, f"missing photos, first one: {missing[0]}"

    print(f"  first three breeds                 {names[:3]}")
    print(f"  last breed                         {names[-1]}")

    images, labels, keys = next(iter(loaders["train"]))
    print("\n  one training batch")
    print(f"    images                           {tuple(images.shape)}")
    print(f"    labels                           {labels.tolist()}")
    print(f"    first photograph                 {keys[0]}")
    print(f"    pixel values                     {images.min():.2f} to {images.max():.2f}")

    # The label must match the breed folder the photo came from.
    folder = keys[0].split("/")[0]
    claimed = names[int(labels[0])]
    print(f"\n  label check                        folder {folder} -> {claimed}")
    assert folder.split("-", 1)[1].replace("_", " ").title() == claimed, \
        "the label does not match the breed folder"

    clean = build_loaders(batch_size=8, num_workers=0, exclude_leaked=True)[2]
    print(f"\n  full test set                      {summary['test_photographs']}")
    print(f"  clean test set                     {clean['test_photographs']} "
          f"({clean['leaked_test_photographs_excluded']} leaked photos excluded)")

    print("\nSelf-test passed.")


if __name__ == "__main__":
    main()
