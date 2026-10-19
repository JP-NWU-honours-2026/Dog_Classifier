"""
train.py - Two-Phase Transfer Learning Pipeline for JP's 120-Breed EfficientNetV2.

Follows the standard Deep Learning Transfer Learning pattern:
  Phase 1: Feature Extraction (Frozen Backbone)
           - Only the custom 120-breed classifier head is trained.
           - Backbone weights remain frozen to protect learned ImageNet features.
           - Optimizer: AdamW with higher learning rate (e.g., lr=3e-4).
  Phase 2: Fine-Tuning (Upper Stages Unfrozen)
           - Upper stages of the backbone are unfrozen (default: Stage 6 & 7).
           - Lower stages (Stages 0-5) remain permanently frozen to preserve basic edge/texture features.
           - Optimizer re-initialized with lower learning rate (e.g., lr=1e-5).

Base Weights & Checkpoint Lifecycle:
  - Fresh Base Weights (Default): Every run naturally starts from pristine ImageNet-1K pretrained weights
    cached locally by PyTorch. No manual reset command is needed.
  - Resume / Fine-Tune Only: Use --resume best_model.pth --phase2_only to continue directly from an
    existing checkpoint without re-running Phase 1.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

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
from model import get_model, freeze_backbone, unfreeze_backbone, get_parameter_summary


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
    """
    Plots and saves publication-grade dual training/validation loss and accuracy curves,
    annotating the boundary between Phase 1 (Frozen Backbone) and Phase 2 (Fine-Tuning).
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    head_epochs = history.get("head_epochs", 0)
    fine_tune_epochs = history.get("fine_tune_epochs", 0)
    has_transition = head_epochs > 0 and fine_tune_epochs > 0 and len(epochs) > head_epochs

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)
    fig.patch.set_facecolor("#fafafa")

    # 1. Loss Curve
    ax1.set_facecolor("#ffffff")
    ax1.plot(epochs, history["train_loss"], "o-", color="#2563eb", linewidth=2.2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s--", color="#dc2626", linewidth=2.2, label="Val Loss")
    if has_transition:
        ax1.axvline(x=head_epochs + 0.5, color="#7c3aed", linestyle="--", linewidth=1.8, label="Fine-Tuning Start")
        y_min, y_max = ax1.get_ylim()
        ax1.text(head_epochs / 2.0 + 0.5, y_max * 0.95, "Phase 1: Head Only", ha="center", va="top",
                 fontsize=9, fontweight="bold", color="#1e40af", bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#93c5fd"))
        ax1.text(head_epochs + 0.5 + fine_tune_epochs / 2.0, y_max * 0.95, "Phase 2: Fine-Tuning", ha="center", va="top",
                 fontsize=9, fontweight="bold", color="#6b21a8", bbox=dict(boxstyle="round,pad=0.3", fc="#f3e8ff", ec="#d8b4fe"))

    ax1.set_title("120-Breed Training & Validation Loss", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0")

    # 2. Accuracy Curve
    ax2.set_facecolor("#ffffff")
    ax2.plot(epochs, history["train_acc"], "o-", color="#059669", linewidth=2.2, label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "s--", color="#d97706", linewidth=2.2, label="Val Acc")
    if has_transition:
        ax2.axvline(x=head_epochs + 0.5, color="#7c3aed", linestyle="--", linewidth=1.8, label="Fine-Tuning Start")
        ax2.text(head_epochs / 2.0 + 0.5, 15, "Phase 1: Head Only", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#1e40af", bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#93c5fd"))
        ax2.text(head_epochs + 0.5 + fine_tune_epochs / 2.0, 15, "Phase 2: Fine-Tuning", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#6b21a8", bbox=dict(boxstyle="round,pad=0.3", fc="#f3e8ff", ec="#d8b4fe"))

    ax2.set_title("120-Breed Training & Validation Accuracy", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[train] Training curves saved to: {save_path}")


def run_training(
    head_epochs: int = 2,
    fine_tune_epochs: int = 3,
    head_lr: float = 0.0003,
    fine_tune_lr: float = 1e-5,
    unfreeze_from_stage: int = 6,  # Default: 6 (unfreezes upper stages 6 & 7 only, leaves 0-5 frozen)
    batch_size: int = 32,
    seed: int = 42,
    subset_fraction: float = 1.0,
    quick_test: bool = False,
    resume_checkpoint: Optional[str] = None,
    output_dir: Path = SCRIPT_DIR,
) -> Tuple[Path, dict]:
    """
    Executes the two-phase transfer learning pipeline:
      Phase 1: Frozen backbone (classifier head warmup)
      Phase 2: Fine-tuning upper stages with lower learning rate (lr=1e-5)
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    if quick_test:
        head_epochs = 0 if resume_checkpoint else 1
        fine_tune_epochs = 1
        subset_fraction = 0.05
        mode_desc = "Phase 2 Only (Resume)" if resume_checkpoint else "1 Head Epoch + 1 Fine-Tune Epoch"
        print(f"\n>>> QUICK TEST MODE ACTIVATED: {mode_desc} on 5% data <<<\n")

    total_epochs = head_epochs + fine_tune_epochs

    print(f"=======================================================")
    print(f" ITRI626 JP_Model - 120-Breed Transfer Learning Pipeline")
    print(f" Device: {device} | Mixed Precision (AMP): {use_amp}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f" Phase 1 (Frozen Backbone):    {head_epochs} epochs @ lr={head_lr}")
    print(f" Phase 2 (Upper Stages Fine):  {fine_tune_epochs} epochs @ lr={fine_tune_lr} (unfreeze stage >= {unfreeze_from_stage})")
    print(f" Total Epochs: {total_epochs} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    workspace_root = output_dir.parent
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        workspace_root=workspace_root,
        batch_size=batch_size,
        seed=seed,
        subset_fraction=subset_fraction,
    )

    num_classes = len(class_names)

    # 1. Initialize model with base ImageNet weights
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    best_val_acc = 0.0
    best_checkpoint_path = output_dir / "best_model.pth"

    # Optional: Resume from existing trained checkpoint
    if resume_checkpoint:
        ckpt_p = Path(resume_checkpoint)
        if not ckpt_p.exists():
            if (output_dir / resume_checkpoint).exists():
                ckpt_p = output_dir / resume_checkpoint
            elif (Path.cwd() / resume_checkpoint).exists():
                ckpt_p = Path.cwd() / resume_checkpoint
            else:
                raise FileNotFoundError(f"Checkpoint file not found: {resume_checkpoint}")

        print(f"[train] Resuming weights from checkpoint: {ckpt_p}")
        ckpt = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"[train] Successfully loaded weights! Baseline Val Accuracy: {best_val_acc:.2f}%\n")
    else:
        print("[train] Starting with fresh ImageNet-1K pretrained base weights (clean reset).\n")

    # Label smoothing assists fine-grained multi-class classification
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "model_name": model.name,
        "num_classes": num_classes,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "head_epochs": head_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "total_epochs": total_epochs,
        "head_lr": head_lr,
        "fine_tune_lr": fine_tune_lr,
        "unfreeze_from_stage": unfreeze_from_stage,
        "batch_size": batch_size,
        "seed": seed,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "phase": [],
    }

    start_training_time = time.time()
    current_global_epoch = 0

    # -------------------------------------------------------------
    # PHASE 1: Feature Extraction / Head Training (Backbone Frozen)
    # -------------------------------------------------------------
    if head_epochs > 0:
        print("=" * 60)
        print(" [PHASE 1] Training Classifier Head (Backbone Frozen)")
        print("=" * 60)
        freeze_backbone(model)
        p_summary = get_parameter_summary(model, verbose=True)

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=head_lr,
            weight_decay=1e-2,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=head_epochs, eta_min=head_lr * 0.1)

        for ep in range(1, head_epochs + 1):
            current_global_epoch += 1
            ep_start = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
            scheduler.step()

            ep_elapsed = time.time() - ep_start
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["phase"].append("Head (Frozen)")

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                torch.save({
                    "epoch": current_global_epoch,
                    "phase": "head",
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

            print(f"[Phase 1 - Head] Epoch [{ep:02d}/{head_epochs:02d}] (Total {current_global_epoch:02d}/{total_epochs:02d}) ({ep_elapsed:.1f}s) "
                  f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% "
                  f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.1f}%{best_marker}")

    # -------------------------------------------------------------
    # PHASE 2: Fine-Tuning Upper Stages (Lower Stages Stay Frozen)
    # -------------------------------------------------------------
    if fine_tune_epochs > 0:
        stage_desc = f"Stages >= {unfreeze_from_stage}" if unfreeze_from_stage > 0 else "All Stages"
        print("\n" + "=" * 60)
        print(f" [PHASE 2] Fine-Tuning Upper Stages ({stage_desc}) with Lower LR ({fine_tune_lr})")
        print("=" * 60)
        unfreeze_backbone(model, unfreeze_from_stage=unfreeze_from_stage)
        p_summary = get_parameter_summary(model, verbose=True)

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=fine_tune_lr,
            weight_decay=1e-2,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-7)

        for ep in range(1, fine_tune_epochs + 1):
            current_global_epoch += 1
            ep_start = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
            scheduler.step()

            ep_elapsed = time.time() - ep_start
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["phase"].append("Fine-Tune (Upper Stages)")

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                torch.save({
                    "epoch": current_global_epoch,
                    "phase": "fine_tune",
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

            print(f"[Phase 2 - FineTune] Epoch [{ep:02d}/{fine_tune_epochs:02d}] (Total {current_global_epoch:02d}/{total_epochs:02d}) ({ep_elapsed:.1f}s) "
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
    parser = argparse.ArgumentParser(description="Train JP's 120-breed EfficientNetV2 with two-phase transfer learning.")
    parser.add_argument("--head_epochs", type=int, default=2, help="Number of Phase 1 epochs (frozen backbone head warmup).")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, help="Number of Phase 2 epochs (upper stages fine-tuning).")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs shorthand (splits into 2 head + remaining fine-tune).")
    parser.add_argument("--batch_size", type=int, default=32 if torch.cuda.is_available() else 16, help="Batch size.")
    parser.add_argument("--head_lr", type=float, default=0.0003, help="Phase 1 learning rate for classifier head.")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Phase 2 learning rate for fine-tuning (e.g. 1e-5).")
    parser.add_argument("--unfreeze_stage", type=int, default=6, help="Backbone stage to unfreeze from (default 6: upper stages only; stages 0-5 stay frozen).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of data to train on (e.g. 0.1 for 10%).")
    parser.add_argument("--quick_test", action="store_true", help="Run 1 head epoch + 1 fine-tuning epoch on 5% data subset.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (e.g. best_model.pth) to resume training from.")
    parser.add_argument("--phase2_only", action="store_true", help="Skip Phase 1 head warmup and go directly into Phase 2 fine-tuning.")

    args = parser.parse_args()

    head_eps = 0 if args.phase2_only else args.head_epochs
    fine_tune_eps = args.fine_tune_epochs

    if args.epochs is not None and not args.phase2_only:
        head_eps = min(2, args.epochs)
        fine_tune_eps = max(0, args.epochs - head_eps)

    run_training(
        head_epochs=head_eps,
        fine_tune_epochs=fine_tune_eps,
        head_lr=args.head_lr,
        fine_tune_lr=args.fine_tune_lr,
        unfreeze_from_stage=args.unfreeze_stage,
        batch_size=args.batch_size,
        seed=args.seed,
        subset_fraction=args.subset_fraction,
        quick_test=args.quick_test,
        resume_checkpoint=args.resume,
    )
