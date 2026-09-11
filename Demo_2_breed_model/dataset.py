"""
dataset.py - Dataset preparation and PyTorch Dataset / DataLoader for 2-Breed Dog Classification.

Aligned with ITRI626 project requirements:
- Fixed random seed for complete reproducibility
- Prevents data leakage between train, val, and test sets
- Saves split manifest (JSON)
- Applies realistic data augmentations to training data only
"""

import os
import json
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Default selected breeds
DEFAULT_BREEDS = {
    0: "n02085620-Chihuahua",
    1: "n02110185-Siberian_husky",
}

# Clean readable names for display
BREED_DISPLAY_NAMES = {
    "n02085620-Chihuahua": "Chihuahua",
    "n02110185-Siberian_husky": "Siberian Husky",
}

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224):
    """
    Returns data transforms for training and validation/test.
    Training includes realistic augmentations (rotation, flip, color jitter).
    Validation/test only resizes and normalizes.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_test_transform


def create_or_load_splits(
    dataset_root: Path,
    output_dir: Path,
    breeds: Dict[int, str] = DEFAULT_BREEDS,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, any]:
    """
    Creates or loads existing reproducible train/val/test splits.
    Saves the split manifest to output_dir/split_manifest.json.
    """
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.exists():
        print(f"[dataset] Loading existing split manifest from: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[dataset] Creating new train/val/test split (seed={seed})...")
    random.seed(seed)

    splits = {"train": [], "val": [], "test": []}

    for label_idx, folder_name in breeds.items():
        breed_dir = dataset_root / folder_name
        if not breed_dir.exists():
            raise FileNotFoundError(f"Breed directory not found: {breed_dir}")

        image_files = sorted([
            p for p in breed_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        if len(image_files) == 0:
            raise ValueError(f"No valid image files found in: {breed_dir}")

        # Shuffle deterministically
        shuffled = list(image_files)
        random.shuffle(shuffled)

        total_count = len(shuffled)
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)

        train_imgs = shuffled[:train_end]
        val_imgs = shuffled[train_end:val_end]
        test_imgs = shuffled[val_end:]

        for img_p in train_imgs:
            splits["train"].append({
                "path": str(img_p.resolve()),
                "rel_path": str(img_p.relative_to(dataset_root.parent.parent)),
                "label": label_idx,
                "breed_folder": folder_name,
                "breed_name": BREED_DISPLAY_NAMES.get(folder_name, folder_name),
            })
        for img_p in val_imgs:
            splits["val"].append({
                "path": str(img_p.resolve()),
                "rel_path": str(img_p.relative_to(dataset_root.parent.parent)),
                "label": label_idx,
                "breed_folder": folder_name,
                "breed_name": BREED_DISPLAY_NAMES.get(folder_name, folder_name),
            })
        for img_p in test_imgs:
            splits["test"].append({
                "path": str(img_p.resolve()),
                "rel_path": str(img_p.relative_to(dataset_root.parent.parent)),
                "label": label_idx,
                "breed_folder": folder_name,
                "breed_name": BREED_DISPLAY_NAMES.get(folder_name, folder_name),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "metadata": {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "breeds": breeds,
            "counts": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
                "total": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
            },
        },
        "splits": splits,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[dataset] Split created successfully! Total: {manifest_data['metadata']['counts']['total']}")
    print(f"          Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    print(f"          Saved manifest to: {manifest_path}")

    return manifest_data


class DogBreedDataset(Dataset):
    """
    PyTorch Dataset loading images from split manifest entries.
    Ensures safe RGB conversion.
    """
    def __init__(self, items: List[Dict[str, any]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        item = self.items[idx]
        img_path = item["path"]
        label = item["label"]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, img_path


def get_dataloaders(
    workspace_root: Optional[Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    High-level factory function returning (train_loader, val_loader, test_loader, class_names).
    """
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent

    dataset_root = workspace_root / "archive" / "images" / "Images"
    output_dir = workspace_root / "Demo_2_breed_model"

    manifest = create_or_load_splits(
        dataset_root=dataset_root,
        output_dir=output_dir,
        seed=seed,
    )

    splits = manifest["splits"]
    train_tf, val_tf = get_transforms()

    train_ds = DogBreedDataset(splits["train"], transform=train_tf)
    val_ds = DogBreedDataset(splits["val"], transform=val_tf)
    test_ds = DogBreedDataset(splits["test"], transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = {0: "Chihuahua", 1: "Siberian Husky"}
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        workspace_root=workspace_root,
        batch_size=8,
    )
    print(f"\nVerification:")
    print(f"Class Names: {class_names}")
    images, labels, paths = next(iter(train_loader))
    print(f"Batch shape: images={images.shape}, labels={labels.shape}")
