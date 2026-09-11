"""
evaluate.py - Test evaluation and metrics reporting for 2-Breed Dog Classifier.

Aligns with ITRI626 Rubric 8:
- Held-out test set evaluation
- Accuracy, Precision, Recall, Macro F1, Per-class metrics
- Confusion Matrix visualization
- Error inspection and sample predictions grid (correct vs incorrect)
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure local imports work cleanly
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import get_dataloaders
from models import get_model


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """Plots and saves a styled confusion matrix using pure Matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#ffffff")

    cax = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title("Test Set Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

    ax.set_ylabel("True Label", fontsize=11, labelpad=8)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix plot saved to: {save_path}")


def plot_sample_predictions(samples: list, class_names: dict, save_path: Path, max_samples: int = 8):
    """
    Plots a grid of test predictions highlighting correct vs misclassified images.
    Green title = Correct, Red title = Misclassified.
    """
    samples_to_plot = samples[:max_samples]
    n = len(samples_to_plot)
    if n == 0:
        return

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 4.2 * rows), dpi=150)
    fig.patch.set_facecolor("#fafafa")
    axes = np.array(axes).reshape(-1)

    for i, item in enumerate(samples_to_plot):
        ax = axes[i]
        img = Image.open(item["path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        true_name = class_names.get(item["true"], str(item["true"]))
        pred_name = class_names.get(item["pred"], str(item["pred"]))
        conf = item["conf"] * 100.0

        is_correct = item["true"] == item["pred"]
        color = "#16a34a" if is_correct else "#dc2626"
        status = "Correct" if is_correct else "Misclassified"

        title = f"{status} ({conf:.1f}%)\nTrue: {true_name}\nPred: {pred_name}"
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=color, pad=6)

        # Border styling
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(2.5)

    # Hide remaining empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Sample predictions plot saved to: {save_path}")


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_dir: Path = SCRIPT_DIR,
):
    """Evaluates a saved checkpoint against the held-out test set."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f" ITRI626 Dog Classifier Demo - Test Evaluation")
    print(f" Checkpoint: {checkpoint_path.name} | Device: {device}")
    print(f"=======================================================\n")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "mobilenet_v3")
    class_names = checkpoint.get("class_names", {0: "Chihuahua", 1: "Siberian Husky"})
    # Convert keys to int if json saved as strings
    class_names = {int(k): v for k, v in class_names.items()}

    model = get_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    workspace_root = output_dir.parent
    _, _, test_loader, _ = get_dataloaders(
        workspace_root=workspace_root,
        batch_size=16,
        seed=checkpoint.get("seed", 42),
    )

    all_preds = []
    all_labels = []
    all_confs = []
    sample_items = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            for i in range(labels.size(0)):
                p = preds[i].item()
                t = labels[i].item()
                c = confs[i].item()
                path = paths[i]

                all_preds.append(p)
                all_labels.append(t)
                all_confs.append(c)

                sample_items.append({
                    "path": path,
                    "true": t,
                    "pred": p,
                    "conf": c,
                    "is_correct": (p == t),
                })

    # Separate correct and error samples so we can display both
    errors = [s for s in sample_items if not s["is_correct"]]
    corrects = [s for s in sample_items if s["is_correct"]]
    # Prioritize showing errors for rubric error analysis
    display_samples = (errors + corrects)[:12]

    # Metrics calculation
    accuracy = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    p_per, r_per, f1_per, support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

    class_names_list = [class_names[i] for i in sorted(class_names.keys())]
    cm = confusion_matrix(all_labels, all_preds)

    # Print Report
    print("-------------------------------------------------------")
    print(f" Test Set Performance Metrics ({len(all_labels)} images)")
    print("-------------------------------------------------------")
    print(f" Test Accuracy:      {accuracy * 100:.2f}%")
    print(f" Macro Precision:    {p_macro * 100:.2f}%")
    print(f" Macro Recall:       {r_macro * 100:.2f}%")
    print(f" Macro F1-Score:     {f1_macro * 100:.2f}%")
    print("-------------------------------------------------------")
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for idx, name in enumerate(class_names_list):
        print(f"{name:<20} | {p_per[idx]*100:9.1f}% | {r_per[idx]*100:9.1f}% | {f1_per[idx]*100:9.1f}% | {support[idx]:<8}")
    print("-------------------------------------------------------\n")

    # Plots
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names_list, cm_path)

    samples_path = output_dir / "sample_predictions.png"
    plot_sample_predictions(display_samples, class_names, samples_path, max_samples=8)

    # Save metrics JSON
    metrics_data = {
        "model_name": checkpoint.get("model_display_name", model_name),
        "test_images_count": len(all_labels),
        "overall": {
            "accuracy": float(accuracy),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
        },
        "per_class": {
            class_names[i]: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f1_per[i]),
                "support": int(support[i]),
            }
            for i in range(len(class_names_list))
        },
        "confusion_matrix": cm.tolist(),
    }

    metrics_json_path = output_dir / "evaluation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[evaluate] Evaluation summary saved to: {metrics_json_path}\n")

    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 2-breed dog classifier.")
    parser.add_argument("--checkpoint", type=str, default=str(SCRIPT_DIR / "best_model.pth"),
                        help="Path to trained model checkpoint.")
    args = parser.parse_args()

    evaluate_checkpoint(checkpoint_path=Path(args.checkpoint))
