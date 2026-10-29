"""
train.py - JP's two-phase transfer learning pipeline, applied to Swin-T.

This is a copy of JP_Model/train.py with only the model labels changed, so the
training recipe (2 head epochs at 3e-4 with the backbone frozen, then 3
fine-tune epochs at 1e-5 with features[6:] unfrozen, AdamW wd 0.01, cosine
schedules, label smoothing 0.1, batch 32, AMP, seed 42) and the logging
format (run_config.json, metrics.csv, history.json, training_curves.png) are
identical across the group. dataset.py, model.py and evaluate.py in this folder are JP's files too,
with Swin-T in place of EfficientNetV2-S.

Runs, from the repository root:
    python Sulaiman_Model/train.py --run-name unclean_transfer --manifest JP_Model/split_manifest.json
    python Sulaiman_Model/train.py --run-name simplified_transfer --manifest JP_Model/split_manifest_simplified.json

Execution Chain & Logging Architecture:
  1. Creates results/runs/<run_name>/ folder.
  2. Writes run_config.json containing hardware, parameters, hyperparameters, and dataset metadata.
  3. Executes Phase 1 (Classifier Head Warmup, frozen backbone) and Phase 2 (Fine-tuning upper stages).
  4. Streams real-time epoch metrics into metrics.csv:
     (epoch, phase, learning_rate, train_loss, train_accuracy, val_loss, val_accuracy, val_top5_accuracy, val_macro_f1, seconds, best_so_far)
  5. Checkpoints best model weights to best_model.pth based on peak validation accuracy.
  6. Generates publication-grade training_curves.png annotating Phase 1 and Phase 2.
"""

import os
import sys
import csv
import json
import time
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Ensure local imports work cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import get_dataloaders
from model import get_model, freeze_backbone, unfreeze_backbone, get_parameter_summary

RUNS_DIR = SCRIPT_DIR / "results" / "runs"


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
    """Evaluates the model on validation set and returns (loss, top1_acc, top5_acc, macro_f1)."""
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_preds = []
    all_labels = []

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

            # Top-1 and Top-5 calculations
            _, top5_preds = outputs.topk(5, 1, True, True)
            correct_top1 += (top5_preds[:, 0] == labels).sum().item()
            correct_top5 += (top5_preds == labels.view(-1, 1)).sum().item()
            total += labels.size(0)

            all_preds.extend(top5_preds[:, 0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = (correct_top1 / total) * 100.0
    epoch_top5 = (correct_top5 / total) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_top5, macro_f1


def plot_training_curves(history: dict, save_path: Path):
    """
    Plots and saves dual training/validation loss and accuracy curves,
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
    ax2.plot(epochs, history["val_acc"], "s--", color="#d97706", linewidth=2.2, label="Val Acc (Top-1)")
    if "val_top5" in history and len(history["val_top5"]) == len(epochs):
        ax2.plot(epochs, history["val_top5"], "^:", color="#7c3aed", linewidth=1.8, label="Val Acc (Top-5)")

    if has_transition:
        ax2.axvline(x=head_epochs + 0.5, color="#7c3aed", linestyle="--", linewidth=1.8)
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
    run_name: str = "unclean_transfer",
    manifest_path: Optional[Path] = None,
    head_epochs: int = 2,
    fine_tune_epochs: int = 3,
    head_lr: float = 0.0003,
    fine_tune_lr: float = 1e-5,
    unfreeze_from_stage: int = 6,
    batch_size: int = 32,
    seed: int = 42,
    subset_fraction: float = 1.0,
    quick_test: bool = False,
    resume_checkpoint: Optional[str] = None,
) -> Tuple[Path, dict]:
    """
    Executes the two-phase transfer learning pipeline with automatic run directory management,
    run_config.json logging, real-time metrics.csv streaming, and curves generation.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")

    if quick_test:
        head_epochs = 0 if resume_checkpoint else 1
        fine_tune_epochs = 1
        subset_fraction = 0.05
        run_name = f"{run_name}_quick"
        print(f"\n>>> QUICK TEST MODE ACTIVATED on 5% data <<<\n")

    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = head_epochs + fine_tune_epochs

    print(f"=======================================================")
    print(f" ITRI626 Sulaiman_Model (Swin-T) - Training Run: {run_name}")
    print(f" Output Directory: {output_dir}")
    print(f" Device: {device} | Mixed Precision (AMP): {use_amp}")
    if device.type == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f" Manifest: {manifest_path or 'split_manifest.json'}")
    print(f" Phase 1 (Frozen Backbone):    {head_epochs} epochs @ lr={head_lr}")
    print(f" Phase 2 (Upper Stages Fine):  {fine_tune_epochs} epochs @ lr={fine_tune_lr} (unfreeze stage >= {unfreeze_from_stage})")
    print(f" Total Epochs: {total_epochs} | Batch Size: {batch_size}")
    print(f"=======================================================\n")

    workspace_root = SCRIPT_DIR.parent
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        workspace_root=workspace_root,
        manifest_path=manifest_path,
        batch_size=batch_size,
        seed=seed,
        subset_fraction=subset_fraction,
    )

    num_classes = len(class_names)

    # 1. Initialize model with base ImageNet weights
    model = get_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # 2. Log run_config.json before training begins
    p_summary = get_parameter_summary(model, verbose=False)
    run_config = {
        "run_name": run_name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "Swin-T (Pretrained Transfer Learning)",
        "architecture": "swin_t",
        "pretrained_weights": "ImageNet-1K (Swin_T_Weights.IMAGENET1K_V1)",
        "total_parameters": p_summary["total_params"],
        "trainable_parameters_initial": p_summary["trainable_params"],
        "manifest_path": str(manifest_path) if manifest_path else "JP_Model/split_manifest.json",
        "classes": num_classes,
        "head_epochs": head_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "total_epochs": total_epochs,
        "head_lr": head_lr,
        "fine_tune_lr": fine_tune_lr,
        "unfreeze_from_stage": unfreeze_from_stage,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if device.type == "cuda" else 0,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"[train] Run config saved to: {output_dir / 'run_config.json'}")

    # 3. Setup metrics.csv file
    metrics_csv_path = output_dir / "metrics.csv"
    csv_fields = [
        "epoch", "phase", "learning_rate", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_top5_accuracy", "val_macro_f1",
        "seconds", "best_so_far"
    ]
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    best_val_acc = 0.0
    best_checkpoint_path = output_dir / "best_model.pth"

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "run_name": run_name,
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
        "val_top5": [],
        "val_macro_f1": [],
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
            curr_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            val_loss, val_acc, val_top5, val_f1 = validate(model, val_loader, criterion, device, use_amp)
            scheduler.step()

            ep_elapsed = time.time() - ep_start
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top5"].append(val_top5)
            history["val_macro_f1"].append(val_f1)
            history["phase"].append("Head (Frozen)")

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                torch.save({
                    "epoch": current_global_epoch,
                    "phase": "head",
                    "model_name": "swin_t",
                    "model_display_name": model.name,
                    "num_classes": num_classes,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_top5": val_top5,
                    "val_macro_f1": val_f1,
                    "class_names": class_names,
                    "seed": seed,
                    "run_name": run_name,
                }, best_checkpoint_path)
                best_marker = " [* BEST SAVED]"
            else:
                best_marker = ""

            # Stream to metrics.csv immediately
            with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow({
                    "epoch": current_global_epoch,
                    "phase": "Phase 1 (Head)",
                    "learning_rate": curr_lr,
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 2),
                    "val_top5_accuracy": round(val_top5, 2),
                    "val_macro_f1": round(val_f1, 4),
                    "seconds": round(ep_elapsed, 1),
                    "best_so_far": round(best_val_acc, 2),
                })

            print(f"[Phase 1 - Head] Epoch [{ep:02d}/{head_epochs:02d}] (Total {current_global_epoch:02d}/{total_epochs:02d}) ({ep_elapsed:.1f}s) "
                  f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% "
                  f"| Val Acc: {val_acc:5.1f}% | Top-5: {val_top5:5.1f}% | Macro F1: {val_f1:.4f}{best_marker}")

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
            curr_lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            val_loss, val_acc, val_top5, val_f1 = validate(model, val_loader, criterion, device, use_amp)
            scheduler.step()

            ep_elapsed = time.time() - ep_start
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top5"].append(val_top5)
            history["val_macro_f1"].append(val_f1)
            history["phase"].append("Fine-Tune (Upper Stages)")

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                torch.save({
                    "epoch": current_global_epoch,
                    "phase": "fine_tune",
                    "model_name": "swin_t",
                    "model_display_name": model.name,
                    "num_classes": num_classes,
                    "state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_top5": val_top5,
                    "val_macro_f1": val_f1,
                    "class_names": class_names,
                    "seed": seed,
                    "run_name": run_name,
                }, best_checkpoint_path)
                best_marker = " [* BEST SAVED]"
            else:
                best_marker = ""

            # Stream to metrics.csv immediately
            with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow({
                    "epoch": current_global_epoch,
                    "phase": "Phase 2 (Fine-Tune)",
                    "learning_rate": curr_lr,
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 2),
                    "val_top5_accuracy": round(val_top5, 2),
                    "val_macro_f1": round(val_f1, 4),
                    "seconds": round(ep_elapsed, 1),
                    "best_so_far": round(best_val_acc, 2),
                })

            print(f"[Phase 2 - FineTune] Epoch [{ep:02d}/{fine_tune_epochs:02d}] (Total {current_global_epoch:02d}/{total_epochs:02d}) ({ep_elapsed:.1f}s) "
                  f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.1f}% "
                  f"| Val Acc: {val_acc:5.1f}% | Top-5: {val_top5:5.1f}% | Macro F1: {val_f1:.4f}{best_marker}")

    total_training_time = time.time() - start_training_time
    history["total_training_time_seconds"] = round(total_training_time, 1)
    history["best_val_accuracy"] = round(best_val_acc, 2)

    print(f"\n[train] Training complete in {total_training_time/60:.1f} minutes! Best Val Accuracy: {best_val_acc:.2f}%")

    # Save history json
    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    curves_path = output_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    # Also update run_config with final training time and best val acc
    run_config["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_config["duration_seconds"] = round(total_training_time, 1)
    run_config["duration_minutes"] = round(total_training_time / 60.0, 2)
    run_config["best_val_accuracy"] = round(best_val_acc, 2)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    return best_checkpoint_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin-T with JP's two-phase transfer learning.")
    parser.add_argument("--run-name", type=str, default="unclean_transfer", help="Folder name under results/runs/ for this run.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to split manifest JSON (e.g. split_manifest_clean.json).")
    parser.add_argument("--head_epochs", type=int, default=2, help="Number of Phase 1 epochs (classifier head warmup).")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, help="Number of Phase 2 epochs (upper stages fine-tuning).")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs shorthand (splits into 2 head + remaining fine-tune).")
    parser.add_argument("--batch_size", type=int, default=32 if torch.cuda.is_available() else 16, help="Batch size.")
    parser.add_argument("--head_lr", type=float, default=0.0003, help="Phase 1 learning rate for classifier head.")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Phase 2 learning rate for fine-tuning.")
    parser.add_argument("--unfreeze_stage", type=int, default=6, help="Backbone stage to unfreeze from (default 6: upper stages 6 & 7).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--subset_fraction", type=float, default=1.0, help="Fraction of data to train on.")
    parser.add_argument("--quick_test", action="store_true", help="Run 1 head epoch + 1 fine-tuning epoch on 5%% data subset.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--phase2_only", action="store_true", help="Skip Phase 1 head warmup and go directly into Phase 2.")

    args = parser.parse_args()

    manifest_p = Path(args.manifest) if args.manifest else None

    head_eps = 0 if args.phase2_only else args.head_epochs
    fine_tune_eps = args.fine_tune_epochs

    if args.epochs is not None and not args.phase2_only:
        head_eps = min(2, args.epochs)
        fine_tune_eps = max(0, args.epochs - head_eps)

    run_training(
        run_name=args.run_name,
        manifest_path=manifest_p,
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
