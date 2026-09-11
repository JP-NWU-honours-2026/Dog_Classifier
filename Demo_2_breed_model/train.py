"""
train.py - Training pipeline for 2-Breed Dog Classifier.

Key features (ITRI626 compliant):
- Device selection (CUDA / CPU)
- Reproducible random seed
- Validation loss and accuracy tracking per epoch
- Best model checkpoint saving (best_model.pth)
- Generates publication-ready training loss and accuracy curve plots
- Saves training history to history.json
"""

import os
import sys
import json
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
from models import get_model


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch and returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
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


def validate(model, dataloader, criterion, device):
    """Evaluates the model on validation set and returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor("#fafafa")

    # 1. Loss Curve
    ax1.set_facecolor("#ffffff")
    ax1.plot(epochs, history["train_loss"], "o-", color="#2563eb", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#dc2626", linewidth=2, label="Val Loss")
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # 2. Accuracy Curve
    ax2.set_facecolor("#ffffff")
    ax2.plot(epochs, history["train_acc"], "o-", color="#059669", linewidth=2, label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "s--", color="#d97706", linewidth=2, label="Val Acc")
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold", pad=12)
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
    model_name: str = "efficientnet_v2",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.0003,
    seed: int = 42,
    output_dir: Path = SCRIPT_DIR,
):
    """Full training pipeline execution."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" ITRI626 Dog Classifier Demo - Training Pipeline")
    print(f" Model: {model_name} | Epochs: {epochs} | Batch: {batch_size} | Device: {device}")
    print(f"=======================================================\n")

    workspace_root = output_dir.parent
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        workspace_root=workspace_root,
        batch_size=batch_size,
        seed=seed,
    )

    model = get_model(model_name=model_name, num_classes=2, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {
        "model_name": model.name,
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

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_display_name": model.name,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "class_names": class_names,
                "seed": seed,
            }, best_checkpoint_path)
            best_marker = " [* BEST SAVED]"
        else:
            best_marker = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%{best_marker}")

    # Save history json
    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[train] Training history saved to: {history_path}")

    # Plot training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    print(f"[train] Training complete! Best Val Accuracy: {best_val_acc:.2f}%\n")
    return best_checkpoint_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2-breed dog classifier.")
    parser.add_argument("--model", type=str, default="efficientnet_v2",
                        choices=["efficientnet_v2", "resnet18", "baseline", "mobilenet_v3"],
                        help="Model architecture to train.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    # Slightly higher learning rate for custom baseline CNN
    learning_rate = args.lr if args.model != "baseline" else 0.001

    run_training(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=learning_rate,
        seed=args.seed,
    )
