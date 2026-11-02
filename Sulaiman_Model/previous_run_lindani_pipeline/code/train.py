"""
train.py  (Sulaiman_Model, Swin Transformer)

Step 3: fine-tunes the pretrained Swin-T on the shared split.

Built from Lindani's Lindani_Model/train.py, so both models are trained, logged
and judged the same way (same epoch loop, same metrics, same early-stopping
rule on validation macro F1, same run_config.json record). What changes is
only what has to change for a PRETRAINED TRANSFORMER instead of a CNN trained
from scratch:

| Setting        | Lindani (scratch CNN) | Sulaiman (pretrained Swin-T)          |
|----------------|-----------------------|---------------------------------------|
| Learning rate  | 1e-3                  | 1e-4  (small, so pretrained knowledge |
|                |                       |        is adjusted, not wiped out)    |
| Warm-up        | none                  | 1 epoch, LR rises from ~0 to 1e-4     |
| Weight decay   | 1e-4                  | 0.05  (standard for Swin)             |
| Batch size     | 64                    | 32    (fits in the RTX 4060's 8 GB)   |
| Epochs         | up to 60              | up to 15 (pretrained converges fast)  |
| Patience       | 5                     | 4                                     |
| Grad clipping  | none                  | max norm 1.0 (keeps transformer       |
|                |                       |        updates stable)                |

Held IDENTICAL to the other two models: the split, 224 x 224 input, the
augmentation, ImageNet normalisation, seed 42, the loss (cross entropy), and
the metrics.

The test pile is never touched here. Training uses the train pile, progress
is measured on the validation pile only. The test pile is scored once, at the
end, by evaluate.py (step 5).

Usage, from the repository root:

    python Sulaiman_Model/train.py --quick     proves everything runs, about a minute
    python Sulaiman_Model/train.py             the real run

If the GPU runs out of memory, add --batch-size 16.

Writes to Sulaiman_Model/results/runs/<name>/:
    run_config.json         every setting, library version and the GPU, for the report
    metrics.csv             one row per epoch, the source of the training curves
    best_model.pth          weights from the best epoch (ignored by git, too big)
    val_confusion_best.csv  validation confusion matrix at the best epoch

RESULT OF THE MAIN RUN (17 October 2026), for quick reference
---------------------------------------------------------------
Command        python Sulaiman_Model/train.py   (all defaults above)
Saved in       Sulaiman_Model/results/runs/swin_t/
Hardware       NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), about 71 s per epoch
Total time     19.0 minutes, 15 of 15 epochs (cap reached, no early stop)
Best epoch     15 (weights saved in best_model.pth)

VALIDATION set (3,025 photos), best epoch:
  accuracy     86.61 per cent
  macro F1     0.8644
  (Lindani's scratch CNN for comparison: 50.51 per cent, macro F1 0.4969)

Validation macro F1 by epoch:
  1: 0.757   5: 0.839   10: 0.860   12: 0.863   15: 0.864
  Flat from epoch 12, so the cap was reached but the model was no longer
  meaningfully improving. More epochs were judged not worth it.

Things to say in the report:
  - Epoch 1 already reached 77 per cent: the pretrained backbone already
    knew what dogs look like, only the final breed decision had to be learned.
  - Final training accuracy 97.72 per cent against 86.61 per cent validation:
    mild overfitting. Validation loss was lowest at epoch 4 (0.513) and crept
    up to 0.533, while accuracy kept rising: the model got more answers right
    but grew overconfident on the ones it got wrong.
  - These are VALIDATION scores. The final reported scores come from the
    test set, scored once by evaluate.py.
"""

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from dataset import INPUT_SIZE, REPO_ROOT, SULAIMAN_DIR, build_loaders
from model import WEIGHTS, build_model, count_parameters

RUNS_DIR = SULAIMAN_DIR / "results" / "runs"
METRIC_FIELDS = [
    "epoch", "learning_rate", "train_loss", "train_accuracy", "train_macro_f1",
    "val_loss", "val_accuracy", "val_macro_f1", "seconds", "best_so_far",
]


def set_seed(seed, deterministic=False):
    """
    Fixes the random starting point so a run can be repeated.

    Honest limitation for the report: GPU arithmetic is still not bit-for-bit
    identical between runs, because some CUDA operations add numbers in a
    non-fixed order. --deterministic forces the slower repeatable versions.
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
    Accuracy and macro-averaged precision, recall and F1 from a confusion matrix
    (rows = true breed, columns = predicted breed). Copied from Lindani's
    train.py so both models are scored by identical code.

    Macro = every breed counts equally, however many photos it has.
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


def warmup_cosine(total_steps, warmup_steps):
    """
    Learning-rate shape, as a multiplier of the base rate, updated every batch:

      - warm-up: rises in a straight line from near 0 to 1 over the first epoch.
        The new head starts random, so its first updates are large and noisy.
        Starting gently stops that noise from damaging the pretrained backbone.
      - cosine decay: then falls smoothly from 1 to 0 by the last batch.
        Big steps early, fine adjustments at the end.
    """
    def multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return multiplier


def run_epoch(model, loader, criterion, device, num_classes, optimiser=None,
              scheduler=None, scaler=None, clip=None, limit_batches=None,
              log_every=50, label=""):
    """
    One pass over one pile. Trains when an optimiser is given, only scores otherwise.

    The four training steps per batch: push the photos through the model, measure
    how wrong it was (the loss), work out which way every weight should move
    (backward), take that step.
    """
    training = optimiser is not None
    model.train(training)       # also switches dropout on in training, off in scoring

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    running_loss, seen, started = 0.0, 0, time.time()
    batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))
    use_amp = scaler is not None and scaler.is_enabled()

    for index, (images, labels, _) in enumerate(loader, 1):
        if limit_batches is not None and index > limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            # Mixed precision: most maths in 16-bit (faster, less GPU memory),
            # weights kept in 32-bit.
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                scores = model(images)
                loss = criterion(scores, labels)

        if training:
            optimiser.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if clip:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimiser.step()
            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        predictions = scores.argmax(dim=1)
        pairs = labels.detach().cpu() * num_classes + predictions.detach().cpu()
        confusion += torch.bincount(pairs, minlength=num_classes ** 2).reshape(
            num_classes, num_classes)

        if index % log_every == 0 or index == batches:
            print(f"    {label} batch {index}/{batches}  "
                  f"loss {running_loss / seen:.3f}  {time.time() - started:.0f}s", flush=True)

    results = metrics_from_confusion(confusion)
    results["loss"] = running_loss / max(seen, 1)
    results["seconds"] = time.time() - started
    return results, confusion


def environment_record():
    """Hardware and library versions, which the brief requires in the report."""
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
    except ImportError:
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
    parser = argparse.ArgumentParser(description="Fine-tune pretrained Swin-T on the shared split.")
    parser.add_argument("--name", default="swin_t", help="folder name for this run")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping max norm, 0 = off")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="must be identical across all three models (Lindani uses 0.0)")
    parser.add_argument("--patience", type=int, default=4,
                        help="epochs without a better validation macro F1 before stopping")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="turn off mixed precision")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run folder")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 epochs over 8 batches each")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs, args.name = 2, f"{args.name}_quick"

    output_dir = RUNS_DIR / args.name
    if output_dir.exists() and not args.overwrite:
        print(f"{output_dir} already exists. Use --overwrite to replace it, "
              f"or --name to give this run its own folder.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    limit = 8 if args.quick else None

    loaders, class_names, data_summary = build_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    steps_per_epoch = limit or len(loaders["train"])
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(steps_per_epoch * args.warmup_epochs))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, warmup_cosine(total_steps, warmup_steps))

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Written before training starts, so even a crashed run documents itself.
    config = {
        "run_name": args.name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "Swin-T (Swin Transformer Tiny), torchvision swin_t",
        "pretrained_weights": str(WEIGHTS),
        "transfer_learning": "new head Dropout + Linear(768 -> 120); whole network "
                             "fine-tuned, nothing frozen",
        "total_parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model),
        "input_size": INPUT_SIZE,
        "classes": num_classes,
        "optimiser": "AdamW",
        "learning_rate": args.lr,
        "schedule": f"linear warm-up over {args.warmup_epochs} epoch(s), then cosine "
                    f"decay to 0, updated every batch",
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "gradient_clipping_max_norm": args.clip or None,
        "loss": "CrossEntropyLoss",
        "label_smoothing": args.label_smoothing,
        "epochs_planned": args.epochs,
        "early_stopping": f"validation macro F1, patience {args.patience}",
        "mixed_precision": use_amp,
        "gradient_accumulation": "none",
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
          f"{data_summary['train_photographs']:,} training photographs, {num_classes} breeds")
    if device.type == "cpu":
        print("No GPU found. This will be very slow; use --quick only.")
    print()

    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=METRIC_FIELDS).writeheader()

    best_f1, best_epoch, epochs_without_gain, started = -1.0, 0, 0, time.time()

    for epoch in range(1, args.epochs + 1):
        learning_rate = optimiser.param_groups[0]["lr"]
        print(f"  epoch {epoch}/{args.epochs}  learning rate at start {learning_rate:.2e}")

        train_results, _ = run_epoch(
            model, loaders["train"], criterion, device, num_classes,
            optimiser=optimiser, scheduler=scheduler, scaler=scaler, clip=args.clip,
            limit_batches=limit, label="train")
        val_results, val_confusion = run_epoch(
            model, loaders["val"], criterion, device, num_classes,
            scaler=scaler, limit_batches=limit, label="val")

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
              + ("   best so far" if improved else f"   no gain for {epochs_without_gain}"))
        if epoch == 1 and device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"    peak GPU memory {peak:.1f} GB")
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
        print("Note: the epoch cap was reached, not early stopping, so the model may "
              "still have been improving. State this in the report.")


if __name__ == "__main__":
    main()
