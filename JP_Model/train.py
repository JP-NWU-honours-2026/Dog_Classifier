"""
train.py - Training and fine-tuning pipeline for JP's 120-Breed EfficientNetV2 Model.

Key features (ITRI626 compliant):
- Automatic GPU detection (NVIDIA GeForce RTX 4050)
- Mixed precision training (AMP) on CUDA for maximum performance
- Fixed random seed for full reproducibility (Rubric 7.1)
- Checkpoints best model (best_model.pth) based on validation accuracy
- Plots publication-ready training loss and accuracy curves (training_curves.png)
- Saves epoch metrics history (history.json)
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Ensure local imports work cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import get_dataloaders
from model import get_model


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp):
    """Trains the model for one epoch and returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, use_amp):
    """Evaluates the model on validation set and returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def plot_training_curves(history: dict, save_path: Path):
    """Plots and saves dual training/validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor("#fafafa")

    # 1. Loss Curve
    ax1.set_facecolor("#ffffff")
    ax1.plot(epochs, history["train_loss"], "o-", color="#2563eb", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#dc2626", linewidth=2, label="Val Loss")
    ax1.set_title("120-Breed Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # 2. Accuracy Curve
    ax2.set_facecolor("#ffffff")
    ax2.plot(epochs, history["train_acc"], "o-", color="#059669", linewidth=2, label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "s--", color="#d97706", linewidth=2, label="Val Acc")
    ax2.set_title("120-Breed Training & Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[train] Training curves saved to: {save_path}")


def run_training(
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.0003,
    seed: int = 42,
    subset_fraction: float = 1.0,
    quick_test: bool = False,
    output_dir: Path = SCRIPT_DIR,
):
    """Executes the full 120-breed training pipeline."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    if quick_test:
        epochs = 1
        subset_fraction = 0.05
        print("\n>>> QUICK TEST MODE ACTIVATED: Running 1 epoch on 5% data subset <<<\n")

    print(f"=======================================================")
    print(f" ITRI626 JP_Model - 120-Breed EfficientNetV2 Training")
    print(f" Device: {device} | Mixed Precision (AMP): {use_amp}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"=======================================================\n")

    workspace_root = output_dir.parent
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        workspace_root=workspace_root,
        batch_size=batch_size,
        seed=seed,
        subset_fraction=subset_fraction,
    )

    num_classes = len(class_names)
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Label smoothing helps multi-class fine-grained classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "model_name": model.name,
        "num_classes": num_classes,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_checkpoint_path = output_dir / "best_model.pth"

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        epoch_elapsed = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": "efficientnet_v2_s",
                "model_display_name": model.name,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "seed": seed,
            }, best_checkpoint_path)
            best_marker = " [* BEST SAVED]"
        else:
            best_marker = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_elapsed:.1f}s) "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%{best_marker}")

    total_training_time = time.time() - start_training_time
    print(f"\n[train] Training complete in {total_training_time/60:.1f} minutes! Best Val Accuracy: {best_val_acc:.2f}%")

    # Save history json
    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[train] History saved to: {history_path}")

    # Plot training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    return best_checkpoint_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JP's 120-breed EfficientNetV2 model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32 if torch.cuda.is_available() else 16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of data to train on (for testing).")
    parser.add_argument("--quick_test", action="store_true", help="Run 1 epoch on 5% data for fast verification.")

    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        quick_test=args.quick_test,
    )
