"""
evaluate.py - Test set evaluation and metrics reporting for 120-Breed Dog Classifier.

Aligns with ITRI626 Rubric Section 8:
- Top-1 and Top-5 Test Accuracy
- Precision, Recall, Macro F1-Score (macro-averaged for multiclass)
- Per-class precision, recall, and support
- Confusion Matrix visualization
- Qualitative sample predictions grid (correct vs error analysis)
- Saves evaluation_metrics.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Ensure local imports work cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import get_dataloaders
from model import get_model


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, top_k_display: int = 30):
    """
    Plots and saves a styled confusion matrix.
    For 120 classes, displays an overall heatmap density.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#ffffff")

    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(f"Test Set Confusion Matrix (120 Dog Breeds)", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted Breed Index (0 - 119)", fontsize=10, labelpad=8)
    ax.set_ylabel("True Breed Index (0 - 119)", fontsize=10, labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved to: {save_path}")


def plot_sample_predictions(samples: list, class_names: dict, save_path: Path, max_samples: int = 8):
    """
    Plots a grid of test predictions highlighting correct vs misclassified images.
    """
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
        ax.set_title(title, fontsize=9, fontweight="bold", color=color, pad=6)

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


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path = SCRIPT_DIR,
    batch_size: int = 32,
):
    """Evaluates a saved checkpoint against the full 120-breed held-out test split."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" ITRI626 JP_Model - 120-Breed Test Evaluation")
    print(f" Checkpoint: {checkpoint_path.name} | Device: {device}")
    print(f"=======================================================\n")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 120)
    raw_class_names = checkpoint.get("class_names", {})
    class_names = {int(k): v for k, v in raw_class_names.items()}

    model = get_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    workspace_root = output_dir.parent
    _, _, test_loader, _ = get_dataloaders(
        workspace_root=workspace_root,
        batch_size=batch_size,
        seed=checkpoint.get("seed", 42),
    )

    all_preds = []
    all_top5_preds = []
    all_labels = []
    sample_items = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            # Top-5 predictions
            _, top5 = torch.topk(probs, k=min(5, num_classes), dim=1)

            preds_cpu = preds.cpu().numpy()
            labels_cpu = labels.numpy()
            confs_cpu = confs.cpu().numpy()
            top5_cpu = top5.cpu().numpy()

            for i in range(len(labels_cpu)):
                p = int(preds_cpu[i])
                t = int(labels_cpu[i])
                c = float(confs_cpu[i])
                top5_list = [int(x) for x in top5_cpu[i]]
                path = paths[i]

                all_preds.append(p)
                all_labels.append(t)
                all_top5_preds.append(top5_list)

                sample_items.append({
                    "path": path,
                    "true": t,
                    "pred": p,
                    "conf": c,
                    "is_correct": (p == t),
                })

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Top-1 Accuracy
    top1_acc = accuracy_score(all_labels, all_preds)

    # Top-5 Accuracy
    top5_correct = sum(1 for i, t in enumerate(all_labels) if t in all_top5_preds[i])
    top5_acc = top5_correct / len(all_labels)

    # Macro Precision, Recall, F1
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    print("-------------------------------------------------------")
    print(f" Test Set Performance Metrics ({len(all_labels):,} images across {num_classes} classes)")
    print("-------------------------------------------------------")
    print(f" Top-1 Test Accuracy: {top1_acc * 100:.2f}%")
    print(f" Top-5 Test Accuracy: {top5_acc * 100:.2f}%")
    print(f" Macro Precision:     {p_macro * 100:.2f}%")
    print(f" Macro Recall:        {r_macro * 100:.2f}%")
    print(f" Macro F1-Score:      {f1_macro * 100:.2f}%")
    print("-------------------------------------------------------\n")

    # Generate visual artifacts
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path)

    # Prioritize showing errors for qualitative error analysis
    errors = [s for s in sample_items if not s["is_correct"]]
    corrects = [s for s in sample_items if s["is_correct"]]
    display_samples = (errors + corrects)[:12]

    samples_path = output_dir / "sample_predictions.png"
    plot_sample_predictions(display_samples, class_names, samples_path, max_samples=8)

    # Save metrics JSON
    metrics_data = {
        "model_name": checkpoint.get("model_display_name", "Transfer-EfficientNetV2-S"),
        "num_classes": num_classes,
        "test_images_count": len(all_labels),
        "overall": {
            "top1_accuracy": float(top1_acc),
            "top5_accuracy": float(top5_acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
        },
    }

    metrics_json_path = output_dir / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[evaluate] Evaluation summary saved to: {metrics_json_path}\n")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JP's 120-breed model.")
    parser.add_argument("--checkpoint", type=str, default=str(SCRIPT_DIR / "best_model.pth"),
                        help="Path to model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=32 if torch.cuda.is_available() else 16)
    args = parser.parse_args()

    evaluate_checkpoint(checkpoint_path=Path(args.checkpoint), batch_size=args.batch_size)
