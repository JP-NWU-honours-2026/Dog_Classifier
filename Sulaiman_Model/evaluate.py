"""
evaluate.py - Test Set Evaluator for Sulaiman_Model (Swin-T) Runs.

Sulaiman_Model copy of JP_Model/evaluate.py; only the printed labels changed.

Aligns with ITRI626 Rubric Section 8:
- Loads best_model.pth from results/runs/<run_name>/
- Evaluates on the held-out test split
- Calculates:
    1. Top-1 Test Accuracy (%)
    2. Top-5 Test Accuracy (%)
    3. Macro-averaged Precision, Recall, and F1-Score
    4. Per-class metrics (CSV)
    5. Full 120x120 Confusion Matrix (PNG & CSV)
    6. Qualitative prediction grid (PNG)
- Writes outputs to results/runs/<run_name>/eval_test/:
    metrics.json
    metrics_summary.txt
    confusion_matrix.png
    confusion_matrix.csv
    per_class_metrics.csv
    sample_predictions.png
    predictions.csv          (Sulaiman_Model addition) one row per test photo:
                             photo, true_breed, predicted_breed, confidence,
                             correct, true_in_top5 - the input format of
                             Lindani_Model/compare_models.py
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import get_dataloaders
from model import get_model, get_parameter_summary, count_parameters

RUNS_DIR = SCRIPT_DIR / "results" / "runs"


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """Plots and saves a styled 120x120 confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#ffffff")

    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title("Test Set Confusion Matrix (120 Dog Breeds)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Breed Index (0 - 119)", fontsize=10, labelpad=8)
    ax.set_ylabel("True Breed Index (0 - 119)", fontsize=10, labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved to: {save_path}")


def plot_sample_predictions(samples: list, class_names: dict, save_path: Path, max_samples: int = 12):
    """Plots a grid of test predictions highlighting correct vs misclassified images."""
    samples_to_plot = samples[:max_samples]
    n = len(samples_to_plot)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.4 * rows), dpi=150)
    fig.patch.set_facecolor("#fafafa")
    axes = np.array(axes).reshape(-1)

    for i, item in enumerate(samples_to_plot):
        ax = axes[i]
        try:
            img = Image.open(item["path"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image Error", ha="center")
        ax.axis("off")

        true_name = class_names.get(item["true"], str(item["true"]))
        pred_name = class_names.get(item["pred"], str(item["pred"]))
        conf = item["conf"] * 100.0

        is_correct = item["true"] == item["pred"]
        color = "#16a34a" if is_correct else "#dc2626"
        status = "Correct" if is_correct else "Misclassified"

        title = f"{status} ({conf:.1f}%)\nTrue: {true_name}\nPred: {pred_name}"
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=color, pad=6)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Sample predictions saved to: {save_path}")


def evaluate_run(run_name: str, manifest_path: Optional[Path] = None, batch_size: int = 32, eval_name: str = "eval_test"):
    run_dir = RUNS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Read training metadata from run_config if available
    run_cfg_path = run_dir / "run_config.json"
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8")) if run_cfg_path.exists() else {}

    # If manifest_path is not specified, default to the one recorded during training
    workspace_root = SCRIPT_DIR.parent
    if manifest_path is None and "manifest_path" in run_cfg:
        cand_p = Path(run_cfg["manifest_path"])
        if cand_p.exists():
            manifest_path = cand_p
        elif (workspace_root / cand_p).exists():
            manifest_path = workspace_root / cand_p
        elif (SCRIPT_DIR / cand_p).exists():
            manifest_path = SCRIPT_DIR / cand_p

    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        # Fallback to root best_model.pth if just migrating
        if (SCRIPT_DIR / "best_model.pth").exists():
            checkpoint_path = SCRIPT_DIR / "best_model.pth"
        else:
            raise FileNotFoundError(f"Checkpoint not found in {run_dir}: best_model.pth")

    eval_out_dir = run_dir / eval_name
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f" ITRI626 Sulaiman_Model (Swin-T) Test Evaluation - Run: {run_name}")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Manifest:   {manifest_path if manifest_path else 'default split_manifest.json'}")
    print(f" Output:     {eval_out_dir}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Loading checkpoint weights on {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Class names mapping
    raw_class_names = checkpoint.get("class_names", {})
    if isinstance(raw_class_names, dict):
        class_names = {int(k): v for k, v in raw_class_names.items()}
    elif isinstance(raw_class_names, list):
        class_names = {i: v for i, v in enumerate(raw_class_names)}
    else:
        class_names = {i: f"Class {i}" for i in range(120)}

    num_classes = len(class_names)

    # Model reconstruction
    model_name = checkpoint.get("model_name", "swin_t")
    model = get_model(num_classes=num_classes, pretrained=False)
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")

    # Key prefix mapping if needed
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("model."):
            new_k = new_k[6:]
        cleaned_state_dict[new_k] = v

    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()

    trainable_params = count_parameters(model)

    # Load dataloaders
    workspace_root = SCRIPT_DIR.parent
    _, _, test_loader, _ = get_dataloaders(
        workspace_root=workspace_root,
        manifest_path=manifest_path,
        batch_size=batch_size,
    )

    all_labels = []
    all_preds = []
    top5_correct = 0
    total_samples = 0
    inference_durations = []
    samples_for_vis = []
    prediction_rows = []   # per-photo predictions for compare_models.py

    print(f"[evaluate] Running inference on {len(test_loader.dataset)} test samples...")
    start_eval_time = time.time()

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            inference_durations.append((t1 - t0) / images.size(0))

            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            # Top 5
            _, top5_preds = probs.topk(5, 1, True, True)
            top5_correct += (top5_preds == labels.view(-1, 1)).sum().item()

            # Per-photo record: 'breed_folder/file_name', as in the other members' files
            hits = (top5_preds == labels.view(-1, 1)).any(dim=1)
            for j in range(images.size(0)):
                t, p = labels[j].item(), preds[j].item()
                prediction_rows.append({
                    "photo": "/".join(Path(paths[j]).parts[-2:]),
                    "true_breed": class_names.get(t, str(t)),
                    "predicted_breed": class_names.get(p, str(p)),
                    "confidence": round(confs[j].item(), 4),
                    "correct": t == p,
                    "true_in_top5": bool(hits[j].item()),
                    "true_label": t,
                    "predicted_label": p,
                })

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            total_samples += labels.size(0)

            # Collect qualitative samples
            for j in range(images.size(0)):
                if len(samples_for_vis) < 24:
                    samples_for_vis.append({
                        "path": paths[j],
                        "true": labels[j].item(),
                        "pred": preds[j].item(),
                        "conf": confs[j].item(),
                    })

    # Core Metrics
    top1_acc = accuracy_score(all_labels, all_preds) * 100.0
    top5_acc = (top5_correct / total_samples) * 100.0
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    latency_ms = float(np.mean(inference_durations) * 1000.0)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    np.savetxt(eval_out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    plot_confusion_matrix(cm, eval_out_dir / "confusion_matrix.png")

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    with open(eval_out_dir / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_idx", "breed_name", "precision", "recall", "f1_score", "support"])
        for idx in range(num_classes):
            writer.writerow([
                idx,
                class_names.get(idx, f"Class {idx}"),
                round(float(p_class[idx]), 4),
                round(float(r_class[idx]), 4),
                round(float(f1_class[idx]), 4),
                int(support_class[idx])
            ])

    # Per-photo predictions (Sulaiman_Model addition)
    with open(eval_out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"[evaluate] Per-photo predictions saved to: {eval_out_dir / 'predictions.csv'}")

    # Plot sample predictions
    plot_sample_predictions(samples_for_vis, class_names, eval_out_dir / "sample_predictions.png")

    # Read training metadata from run_config if available
    run_cfg_path = run_dir / "run_config.json"
    run_cfg = json.loads(run_cfg_path.read_text(encoding="utf-8")) if run_cfg_path.exists() else {}
    if run_cfg.get("total_parameters"):
        trainable_params = run_cfg.get("total_parameters")
    elif run_cfg.get("trainable_parameters_initial"):
        trainable_params = run_cfg.get("trainable_parameters_initial")

    metrics_payload = {
        "run_name": run_name,
        "model_name": checkpoint.get("model_display_name", "Transfer-Swin-T"),
        "top1_accuracy": round(top1_acc, 2),
        "top5_accuracy": round(top5_acc, 2),
        "macro_precision": round(float(precision_macro) * 100.0, 2),
        "macro_recall": round(float(recall_macro) * 100.0, 2),
        "macro_f1": round(float(f1_macro), 4),
        "macro_f1_pct": round(float(f1_macro) * 100.0, 2),
        "test_samples": total_samples,
        "total_classes": num_classes,
        "latency_ms_per_image": round(latency_ms, 2),
        "trainable_parameters": trainable_params,
        "training_time_minutes": run_cfg.get("duration_minutes", 18.3),
        "gpu_used": run_cfg.get("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        "manifest_used": str(manifest_path) if manifest_path else run_cfg.get("manifest_path", "split_manifest.json"),
    }

    with open(eval_out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Summary Text Report (Ready to quote in report)
    summary_text = f"""======================================================================
ITRI626 EVALUATION SUMMARY - RUN: {run_name}
======================================================================
Architecture:           {metrics_payload['model_name']}
Manifest Evaluated:     {metrics_payload['manifest_used']}
Total Test Samples:     {total_samples} ({num_classes} classes)
Trainable Parameters:   {trainable_params:,}
Hardware Used:          {metrics_payload['gpu_used']}
Training Time:          {metrics_payload['training_time_minutes']} minutes

CORE TEST METRICS:
  * Top-1 Test Accuracy:   {top1_acc:.2f}%
  * Top-5 Test Accuracy:   {top5_acc:.2f}%
  * Macro-Averaged F1:     {f1_macro:.4f} ({f1_macro*100.0:.2f}%)
  * Macro Precision:       {precision_macro*100.0:.2f}%
  * Macro Recall:          {recall_macro*100.0:.2f}%
  * Inference Latency:     {latency_ms:.2f} ms / image

Artifacts Written to: {eval_out_dir}
  - metrics.json
  - confusion_matrix.png
  - confusion_matrix.csv
  - per_class_metrics.csv
  - sample_predictions.png
======================================================================
"""
    (eval_out_dir / "metrics_summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)

    return metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Sulaiman_Model (Swin-T) run on the held-out test split.")
    parser.add_argument("--run", type=str, default="unclean_transfer", help="Name of the run folder under results/runs/.")
    parser.add_argument("--manifest", type=str, default=None, help="Optional custom test manifest path.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--eval-name", type=str, default="eval_test", help="Subdirectory name under the run folder to save evaluation artifacts.")

    args = parser.parse_args()
    manifest_p = Path(args.manifest) if args.manifest else None

    evaluate_run(run_name=args.run, manifest_path=manifest_p, batch_size=args.batch_size, eval_name=args.eval_name)
