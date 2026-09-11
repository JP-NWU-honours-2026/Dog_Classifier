"""
dataset.py - Dataset indexing, leakage-free 70/15/15 splitting, and PyTorch DataLoaders
for all 120 dog breeds in the Stanford Dogs Dataset (ITRI626).
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def clean_breed_name(folder_name: str) -> str:
    """
    Converts raw folder names (e.g., 'n02085620-Chihuahua' or 'n02099601-golden_retriever')
    into clean, formatted breed names ('Chihuahua', 'Golden Retriever').
    """
    # Remove WordNet ID prefix (e.g. n02085620-)
    parts = folder_name.split("-", 1)
    name_part = parts[1] if len(parts) > 1 else parts[0]
    return name_part.replace("_", " ").title()


def get_transforms(image_size: int = 224):
    """
    Returns data transforms for training and validation/test.
    Training includes realistic augmentations (rotation, flip, color jitter).
    Validation and test only apply resizing and normalization (Rubric Section 7.2).
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
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
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    validate_images: bool = False,
) -> Dict[str, any]:
    """
    Creates or loads a reproducible 70/15/15 train/val/test split across all 120 breeds.
    Saves the permanent split manifest to output_dir/split_manifest.json.
    """
    manifest_path = output_dir / "split_manifest.json"
    if manifest_path.exists():
        print(f"[dataset] Loading existing 120-breed split manifest from: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[dataset] Scanning all breed folders in: {dataset_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")

    breed_folders = sorted([d.name for d in dataset_root.iterdir() if d.is_dir()])
    total_breeds = len(breed_folders)
    print(f"[dataset] Found {total_breeds} breed directories. Creating reproducible splits (seed={seed})...")

    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}
    classes_map = {}
    corrupted_count = 0

    for class_idx, folder_name in enumerate(breed_folders):
        display_name = clean_breed_name(folder_name)
        classes_map[class_idx] = {
            "folder": folder_name,
            "display_name": display_name,
        }

        breed_dir = dataset_root / folder_name
        image_files = sorted([
            p for p in breed_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        # Optional integrity check
        if validate_images:
            valid_files = []
            for img_p in image_files:
                try:
                    with Image.open(img_p) as im:
                        im.verify()
                    valid_files.append(img_p)
                except Exception:
                    corrupted_count += 1
            image_files = valid_files

        # Deterministic shuffle
        shuffled = list(image_files)
        random.shuffle(shuffled)

        n_total = len(shuffled)
        train_end = int(n_total * train_ratio)
        val_end = train_end + int(n_total * val_ratio)

        train_imgs = shuffled[:train_end]
        val_imgs = shuffled[train_end:val_end]
        test_imgs = shuffled[val_end:]

        for img_p in train_imgs:
            splits["train"].append({
                "path": str(img_p.resolve()),
                "label": class_idx,
                "breed_folder": folder_name,
                "breed_name": display_name,
            })
        for img_p in val_imgs:
            splits["val"].append({
                "path": str(img_p.resolve()),
                "label": class_idx,
                "breed_folder": folder_name,
                "breed_name": display_name,
            })
        for img_p in test_imgs:
            splits["test"].append({
                "path": str(img_p.resolve()),
                "label": class_idx,
                "breed_folder": folder_name,
                "breed_name": display_name,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "metadata": {
            "total_classes": total_breeds,
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "corrupted_images_removed": corrupted_count,
            "counts": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"]),
                "total": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
            },
        },
        "classes": classes_map,
        "splits": splits,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    counts = manifest_data["metadata"]["counts"]
    print(f"[dataset] 120-Breed split created successfully!")
    print(f"          Total Images: {counts['total']:,}")
    print(f"          Train: {counts['train']:,} ({counts['train']/counts['total']*100:.1f}%)")
    print(f"          Val:   {counts['val']:,} ({counts['val']/counts['total']*100:.1f}%)")
    print(f"          Test:  {counts['test']:,} ({counts['test']/counts['total']*100:.1f}%)")
    print(f"          Manifest saved to: {manifest_path}")

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
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    subset_fraction: float = 1.0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str]]:
    """
    High-level factory function returning (train_loader, val_loader, test_loader, class_names).
    Optionally accepts subset_fraction for quick debugging/testing (e.g. 0.05 = 5% of data).
    """
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent

    dataset_root = workspace_root / "archive" / "images" / "Images"
    output_dir = workspace_root / "JP_Model"

    manifest = create_or_load_splits(
        dataset_root=dataset_root,
        output_dir=output_dir,
        seed=seed,
    )

    splits = manifest["splits"]
    train_items = splits["train"]
    val_items = splits["val"]
    test_items = splits["test"]

    if subset_fraction < 1.0:
        random.seed(seed)
        n_train = max(16, int(len(train_items) * subset_fraction))
        n_val = max(16, int(len(val_items) * subset_fraction))
        n_test = max(16, int(len(test_items) * subset_fraction))
        train_items = random.sample(train_items, n_train)
        val_items = random.sample(val_items, n_val)
        test_items = random.sample(test_items, n_test)
        print(f"[dataset] Using {subset_fraction*100:.0f}% subset: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")

    train_tf, val_tf = get_transforms()

    train_ds = DogBreedDataset(train_items, transform=train_tf)
    val_ds = DogBreedDataset(val_items, transform=val_tf)
    test_ds = DogBreedDataset(test_items, transform=val_tf)

    # Pin memory for fast GPU transfers if CUDA is available
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    raw_classes = manifest["classes"]
    class_names = {int(k): v["display_name"] for k, v in raw_classes.items()}

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        workspace_root=workspace_root,
        batch_size=16,
    )
    print(f"\nVerification:")
    print(f"Total Classes: {len(class_names)}")
    print(f"First 5 Classes: {[(k, class_names[k]) for k in sorted(class_names.keys())[:5]]}")
    images, labels, paths = next(iter(train_loader))
    print(f"Sample Batch: images={images.shape}, labels={labels.shape}")
