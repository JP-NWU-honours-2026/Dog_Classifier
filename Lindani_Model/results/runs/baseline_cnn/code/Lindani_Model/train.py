"""
train.py

Trains the baseline CNN on the shared split.

Two rules this file follows deliberately.

**The test set is never touched here.** This script trains on the training pile
and measures progress on the validation pile only. The test pile is scored once,
later, by evaluate.py. If test scores were visible while settings were being
adjusted, the final number would be worthless.

**Everything the specification asks about reproducibility is logged
automatically**, into run_config.json beside the results, rather than
reconstructed from memory at write-up time: input size, batch size, epochs,
early stopping settings, optimiser, learning rate, schedule, weight decay, loss
function, label smoothing, seed, library versions, and the GPU the run used.

Usage, from the repository root:

    python Lindani_Model/train.py --quick          proves the pipeline runs, minutes
    python Lindani_Model/train.py                  the real run, on a GPU

Training on a laptop CPU is not practical. Use Kaggle's free GPU, or the --quick
flag to check the machinery works.

Needs: pip install torch torchvision
"""

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from dataset import INPUT_SIZE, build_loaders
from model import BaselineCNN, count_parameters
from split_check import LINDANI_DIR, REPO_ROOT

RUNS_DIR = LINDANI_DIR / "results" / "runs"
METRIC_FIELDS = [
    "epoch", "learning_rate", "train_loss", "train_accuracy", "train_macro_f1",
    "val_loss", "val_accuracy", "val_macro_f1", "seconds", "best_so_far",
]


def set_seed(seed, deterministic=False):
    """
    Fixes the random starting point so a run can be repeated.

    Honest limitation for the report: this makes the run repeatable, but it does
    not make GPU arithmetic bitwise identical between runs, because some CUDA
    operations add numbers in a non-fixed order. --deterministic forces the
    slower, fully repeatable versions where they exist.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def metrics_from_confusion(matrix):
    """
    Turns a confusion matrix into accuracy and macro averaged precision, recall
    and F1, without needing scikit-learn.

    Rows are the true breed, columns the predicted one, so the diagonal holds
    the correct predictions.

    Macro averaging treats all 120 breeds equally, rather than letting the
    breeds with the most photographs dominate. That matters here: plain accuracy
    can look respectable while the model is hopeless on the rarer breeds.
    """
    matrix = matrix.double()
    correct = matrix.diag()
    predicted = matrix.sum(dim=0)
    actual = matrix.sum(dim=1)
    zero = torch.zeros_like(correct)

    precision = torch.where(predicted > 0, correct / predicted, zero)
    recall = torch.where(actual > 0, correct / actual, zero)
    total = precision + recall
    f1 = torch.where(total > 0, 2 * precision * recall / total, zero)

    return {
        "accuracy": (correct.sum() / matrix.sum()).item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


def run_epoch(model, loader, criterion, device, num_classes, optimiser=None,
              scaler=None, limit_batches=None, log_every=50, label=""):
    """
    One pass over one pile. Training when an optimiser is given, scoring only
    when it is not.

    In training mode the four steps are: push the batch through the model, work
    out how wrong it was, work out which direction every weight should move
    (backward), then take that step.
    """
    training = optimiser is not None
    model.train(training)

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    running_loss, seen, started = 0.0, 0, time.time()
    batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))

    for index, (images, labels, _) in enumerate(loader, 1):
        if limit_batches is not None and index > limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            # Mixed precision: most of the arithmetic runs in half precision,
            # which is faster on a GPU, while the weights stay full precision.
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None
                                    and scaler.is_enabled()):
                scores = model(images)
                loss = criterion(scores, labels)

        if training:
            optimiser.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        predictions = scores.argmax(dim=1)
        pairs = labels.detach().cpu() * num_classes + predictions.detach().cpu()
        confusion += torch.bincount(pairs, minlength=num_classes ** 2).reshape(
            num_classes, num_classes)

        if index % log_every == 0 or index == batches:
            done = time.time() - started
            print(f"    {label} batch {index}/{batches}  "
                  f"loss {running_loss / seen:.3f}  {done:.0f}s", flush=True)

    results = metrics_from_confusion(confusion)
    results["loss"] = running_loss / max(seen, 1)
    results["seconds"] = time.time() - started
    return results, confusion


def environment_record():
    """The hardware and library versions the specification asks us to report."""
    record = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torchvision
        record["torchvision"] = torchvision.__version__
    except ImportError:  # pragma: no cover
        record["torchvision"] = "not installed"
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
        record["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        record["cuda"] = torch.version.cuda
    try:
        record["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        record["git_commit"] = "unknown"
    return record


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline CNN.")
    parser.add_argument("--name", default="baseline_cnn", help="folder name for this run")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="keep identical across all three models")
    parser.add_argument("--patience", type=int, default=5,
                        help="epochs without a better validation macro F1 before stopping")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="slower, but repeatable to the last decimal on GPU")
    parser.add_argument("--no-amp", action="store_true", help="turn off mixed precision")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 epochs over a handful of batches")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs, args.name = 2, f"{args.name}_quick"

    output_dir = RUNS_DIR / args.name
    if output_dir.exists() and not args.overwrite:
        print(f"{output_dir} already exists. Use --overwrite to replace it, or --name "
              f"to give this run its own folder.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit = 8 if args.quick else None

    loaders, class_names, data_summary = build_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    num_classes = len(class_names)

    model = BaselineCNN(num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Written before training starts, so even a run that crashes documents itself.
    config = {
        "run_name": args.name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "BaselineCNN, five convolutional blocks, trained from scratch",
        "pretrained_weights": "none",
        "trainable_parameters": count_parameters(model),
        "input_size": INPUT_SIZE,
        "classes": num_classes,
        "optimiser": "AdamW",
        "learning_rate": args.lr,
        "schedule": f"CosineAnnealingLR over {args.epochs} epochs",
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "loss": "CrossEntropyLoss",
        "label_smoothing": args.label_smoothing,
        "epochs_planned": args.epochs,
        "early_stopping": f"validation macro F1, patience {args.patience}",
        "mixed_precision": use_amp,
        "deterministic_algorithms": args.deterministic,
        "seed": args.seed,
        "device": str(device),
        "augmentation": "training only: horizontal flip, rotation up to 15 degrees, "
                        "mild colour jitter. Identical to the shared pipeline.",
        "data": data_summary,
        "environment": environment_record(),
        "note": "The test pile is not read by this script. It is scored once by "
                "evaluate.py after training is finished.",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Training on {device}"
          + (f" ({config['environment'].get('gpu')})" if device.type == "cuda" else ""))
    print(f"{config['trainable_parameters']:,} trainable parameters, "
          f"{data_summary['train_photographs']:,} training photographs, "
          f"{num_classes} breeds")
    if device.type == "cpu":
        print("No GPU found. This will be slow. Use --quick here and train on Kaggle.")
    print()

    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    best_f1, best_epoch, epochs_without_gain, started = -1.0, 0, 0, time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"  epoch {epoch}/{args.epochs}  learning rate "
              f"{optimiser.param_groups[0]['lr']:.2e}")
        learning_rate = optimiser.param_groups[0]["lr"]

        train_results, _ = run_epoch(
            model, loaders["train"], criterion, device, num_classes,
            optimiser=optimiser, scaler=scaler, limit_batches=limit, label="train")
        val_results, val_confusion = run_epoch(
            model, loaders["val"], criterion, device, num_classes,
            limit_batches=limit, label="val")
        scheduler.step()

        improved = val_results["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, epochs_without_gain = val_results["macro_f1"], epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "val_accuracy": val_results["accuracy"],
                "class_names": class_names,
                "config": config,
            }, output_dir / "best_model.pth")
            np.savetxt(output_dir / "val_confusion_best.csv",
                       val_confusion.numpy(), fmt="%d", delimiter=",")
        else:
            epochs_without_gain += 1

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=METRIC_FIELDS).writerow({
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": round(train_results["loss"], 5),
                "train_accuracy": round(train_results["accuracy"], 5),
                "train_macro_f1": round(train_results["macro_f1"], 5),
                "val_loss": round(val_results["loss"], 5),
                "val_accuracy": round(val_results["accuracy"], 5),
                "val_macro_f1": round(val_results["macro_f1"], 5),
                "seconds": round(train_results["seconds"] + val_results["seconds"], 1),
                "best_so_far": improved,
            })

        print(f"    train loss {train_results['loss']:.3f}  "
              f"accuracy {100 * train_results['accuracy']:.2f}%")
        print(f"    val   loss {val_results['loss']:.3f}  "
              f"accuracy {100 * val_results['accuracy']:.2f}%  "
              f"macro F1 {val_results['macro_f1']:.4f}"
              + ("   best so far" if improved else
                 f"   no gain for {epochs_without_gain}"))
        print()

        if epochs_without_gain >= args.patience:
            print(f"Early stopping: no better validation macro F1 for {args.patience} epochs.")
            break

    minutes = (time.time() - started) / 60
    config.update({
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epochs_run": epoch,
        "stopped_early": epoch < args.epochs,
        "epoch_cap_reached": epoch == args.epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_f1, 5),
        "total_minutes": round(minutes, 1),
    })
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Best epoch {best_epoch}, validation macro F1 {best_f1:.4f}, "
          f"{minutes:.1f} minutes total.")
    print(f"Saved to {output_dir}")
    if config["epoch_cap_reached"]:
        print("Note: the epoch cap was reached rather than early stopping, so the model "
              "may still have been improving. Report this, and consider more epochs.")


if __name__ == "__main__":
    main()
