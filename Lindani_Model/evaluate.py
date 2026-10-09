"""
evaluate.py

Stage 6: scores a trained model and produces the evaluation for the report.

**The rule this script enforces.** By default it scores the VALIDATION set, which
may be used as often as you like. It reads the TEST set only when given --final,
and refuses to score test twice unless told to overwrite. Iterating against test
scores quietly turns the final figure into "how the model does on photographs I
kept adjusting against", which is not what the report claims it is.

So the loop is: evaluate on validation, change something, retrain, evaluate on
validation again, and only once the final model is chosen, run with --final.

Produces, in results/runs/<run>/eval_<split>/:

    metrics_summary.txt      the headline figures, ready to quote
    metrics.json             the same numbers for the record
    per_class_metrics.csv    precision, recall, F1 and support for all 120 breeds
    confusion_matrix.csv     the full 120 by 120 matrix
    top_confusions.csv       the 20 most frequent mistakes, worst first
    predictions.csv          every photograph with its true breed, prediction and confidence
    fig_curves.png           training and validation loss, and validation macro F1
    fig_confusion_worst.png  a readable submatrix of the 15 weakest breeds
    fig_correct.png          correct predictions with confidence
    fig_wrong.png            mistakes, with what it said instead

With --final it also reports the clean test score, which leaves out the test
photographs that have a confirmed duplicate in training, so the leakage measured
in Stage 2 appears as a number rather than a worry.

Usage, from the repository root:

    python Lindani_Model/evaluate.py                      scores validation
    python Lindani_Model/evaluate.py --final              scores test, once

Needs: pip install torch torchvision matplotlib
"""

import argparse
import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import build_loaders, leaked_keys
from dataset_figures import ACCENT, DATA, INK, MUTED, square_tile
from label_audit import wilson_interval
from model import BaselineCNN, count_parameters
from split_check import LINDANI_DIR

RUNS_DIR = LINDANI_DIR / "results" / "runs"
EXAMPLE_TILES, EXAMPLE_COLS = 12, 4
WORST_BREEDS = 15          # size of the readable confusion submatrix
TOP_CONFUSIONS = 20


def load_model(run_dir, device):
    """Loads the best weights saved during training, with the run's own settings."""
    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    config = checkpoint.get("config", {})
    dropout = config.get("dropout", 0.3)
    # Older runs predate the variant setting, so they are one conv per block.
    convs = tuple(config.get("convs_per_block", (1, 1, 1, 1, 1)))

    model = BaselineCNN(num_classes=len(class_names), dropout=dropout, convs_per_block=convs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, checkpoint


def predict(model, loader, device, num_classes):
    """
    Runs every photograph through the model once and records what it said.

    Returns the confusion matrix and a row per photograph: its name, true breed,
    predicted breed and how confident the model was. Dropout is off and no
    gradients are tracked, because nothing is being learned here.
    """
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    rows = []

    with torch.no_grad():
        for images, labels, keys in loader:
            images = images.to(device, non_blocking=True)
            scores = model(images)
            probabilities = torch.softmax(scores, dim=1).cpu()
            confidence, predictions = probabilities.max(dim=1)

            # Top five tells us whether the right answer was at least close.
            top5 = probabilities.topk(min(5, num_classes), dim=1).indices

            pairs = labels * num_classes + predictions
            confusion += torch.bincount(pairs, minlength=num_classes ** 2).reshape(
                num_classes, num_classes)

            for n, key in enumerate(keys):
                rows.append({
                    "photo": key,
                    "true_label": int(labels[n]),
                    "predicted_label": int(predictions[n]),
                    "confidence": round(float(confidence[n]), 4),
                    "correct": int(labels[n]) == int(predictions[n]),
                    "true_in_top5": int(labels[n]) in top5[n].tolist(),
                })

    return confusion, rows


def metrics_from_confusion(confusion):
    """
    Accuracy and the macro averaged scores, worked out from the confusion matrix.

    Precision asks: of the photographs called this breed, how many were. Recall
    asks: of the photographs that were this breed, how many were found. F1
    balances the two. Macro averaging gives all 120 breeds equal weight, so the
    model cannot look good by being right on the breeds with the most
    photographs.
    """
    matrix = confusion.double()
    correct = matrix.diag()
    predicted = matrix.sum(dim=0)
    actual = matrix.sum(dim=1)
    zero = torch.zeros_like(correct)

    precision = torch.where(predicted > 0, correct / predicted, zero)
    recall = torch.where(actual > 0, correct / actual, zero)
    total = precision + recall
    f1 = torch.where(total > 0, 2 * precision * recall / total, zero)

    # Weighted averages count each breed in proportion to how many photographs
    # it has, which is close to accuracy but reported because the brief asks for
    # precision, recall and F1 as well as their macro versions.
    weights = actual / actual.sum()

    return {
        "accuracy": (correct.sum() / matrix.sum()).item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "weighted_precision": (precision * weights).sum().item(),
        "weighted_recall": (recall * weights).sum().item(),
        "weighted_f1": (f1 * weights).sum().item(),
    }, precision.numpy(), recall.numpy(), f1.numpy(), actual.numpy()


def subset_metrics(rows, num_classes, keep):
    """The same numbers for a subset of the photographs, for example the clean test set."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    kept = [row for row in rows if keep(row)]
    for row in kept:
        confusion[row["true_label"], row["predicted_label"]] += 1
    summary, *_ = metrics_from_confusion(confusion)
    summary["photographs"] = len(kept)
    return summary


def figure_curves(run_dir, output_dir):
    """
    The learning history: training and validation loss, and validation macro F1.

    The gap between the two loss lines is overfitting made visible, and the
    marked best epoch shows where early stopping took its weights from.
    """
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    if not history:
        return

    epochs = [int(r["epoch"]) for r in history]
    train_loss = [float(r["train_loss"]) for r in history]
    val_loss = [float(r["val_loss"]) for r in history]
    val_f1 = [float(r["val_macro_f1"]) for r in history]
    best = max(range(len(val_f1)), key=lambda i: val_f1[i])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.plot(epochs, train_loss, color=DATA, label="training")
    ax1.plot(epochs, val_loss, color=ACCENT, label="validation")
    ax1.axvline(epochs[best], color=MUTED, linestyle=":", linewidth=1.2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss. The gap between the lines is overfitting")
    ax1.legend(frameon=False)

    ax2.plot(epochs, val_f1, color=DATA)
    ax2.axvline(epochs[best], color=MUTED, linestyle=":", linewidth=1.2)
    ax2.plot([epochs[best]], [val_f1[best]], "o", color=ACCENT)
    ax2.annotate(f"best epoch {epochs[best]}, {val_f1[best]:.3f}",
                 (epochs[best], val_f1[best]), textcoords="offset points",
                 xytext=(-10, 10), fontsize=8, color=ACCENT, ha="right")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation macro F1")
    ax2.set_title("Validation macro F1, which early stopping watched")

    fig.tight_layout()
    fig.savefig(output_dir / "fig_curves.png")
    plt.close(fig)


def figure_worst_confusion(confusion, names, f1, output_dir):
    """
    A 120 by 120 matrix is unreadable on paper, so this shows the submatrix of
    the weakest breeds by F1, where the mistakes are concentrated.
    """
    worst = sorted(range(len(names)), key=lambda i: f1[i])[:WORST_BREEDS]
    worst = sorted(worst, key=lambda i: names[i])
    block = confusion.numpy()[np.ix_(worst, worst)]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.imshow(block, cmap="Blues")
    ax.set_xticks(range(len(worst)))
    ax.set_yticks(range(len(worst)))
    ax.set_xticklabels([names[i] for i in worst], rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels([names[i] for i in worst], fontsize=7)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("true breed", fontsize=9)
    ax.set_title(f"The {WORST_BREEDS} weakest breeds by F1. The diagonal is correct answers",
                 fontsize=10, color=INK)

    for y in range(len(worst)):
        for x in range(len(worst)):
            if block[y, x]:
                ax.text(x, y, int(block[y, x]), ha="center", va="center", fontsize=6,
                        color="white" if block[y, x] > block.max() * 0.6 else INK)

    fig.tight_layout()
    fig.savefig(output_dir / "fig_confusion_worst.png")
    plt.close(fig)


def figure_examples(rows, names, output_dir, correct, filename, title):
    """
    Photographs the model got right or wrong, with what it said and how sure it
    was. The confident mistakes are the interesting ones for the discussion.
    """
    chosen = [r for r in rows if r["correct"] == correct]
    if not chosen:
        return
    # Most confident first: the confident errors are the ones worth explaining.
    chosen = sorted(chosen, key=lambda r: -r["confidence"])[:EXAMPLE_TILES]
    rows_needed = -(-len(chosen) // EXAMPLE_COLS)

    # constrained_layout keeps the two line captions from landing on the picture
    # above them, which plain tight_layout does not manage with a suptitle.
    fig, axes = plt.subplots(rows_needed, EXAMPLE_COLS, figsize=(13, 4.0 * rows_needed),
                             constrained_layout=True)
    for ax, row in zip(np.array(axes).ravel(), chosen):
        ax.imshow(square_tile(row["photo"]))
        true_name = names[row["true_label"]]
        said = names[row["predicted_label"]]
        caption = (f"{true_name}\n{row['confidence']:.0%} confident" if correct
                   else f"{true_name}\nsaid {said}, {row['confidence']:.0%} confident")
        ax.set_title(caption, fontsize=8, color=INK, pad=3)
    for ax in np.array(axes).ravel():
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.savefig(output_dir / filename)
    plt.close(fig)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline CNN.")
    parser.add_argument("--run", default="baseline_cnn", help="run folder under results/runs")
    parser.add_argument("--final", action="store_true",
                        help="score the TEST set. Only for the final chosen model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing evaluation folder")
    return parser.parse_args()


def main():
    args = parse_args()
    split = "test" if args.final else "val"
    run_dir = RUNS_DIR / args.run
    output_dir = run_dir / f"eval_{split}"

    if not (run_dir / "best_model.pth").is_file():
        print(f"No trained weights at {run_dir / 'best_model.pth'}.")
        return

    if output_dir.exists() and not args.overwrite:
        print(f"{output_dir} already exists.")
        if split == "test":
            print("The test set has already been scored for this run. Scoring it again "
                  "after changing anything is how a test score stops meaning what the "
                  "report says it means. Use --overwrite only if this is a rerun of the "
                  "same unchanged model.")
        else:
            print("Use --overwrite to replace it.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, checkpoint = load_model(run_dir, device)
    num_classes = len(names)

    loaders, loader_names, data_summary = build_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers)
    assert loader_names == names, "the split file and the checkpoint disagree about breeds"

    if split == "test":
        print("Scoring the TEST set. This is the one that counts.\n")
    else:
        print("Scoring the VALIDATION set. The test set stays sealed.\n")

    confusion, rows = predict(model, loaders[split], device, num_classes)
    summary, precision, recall, f1, support = metrics_from_confusion(confusion)

    # ----- tables -----
    for row in rows:
        row["true_breed"] = names[row["true_label"]]
        row["predicted_breed"] = names[row["predicted_label"]]
    write_csv(output_dir / "predictions.csv",
              ["photo", "true_breed", "predicted_breed", "confidence", "correct",
               "true_in_top5", "true_label", "predicted_label"], rows)

    per_class = [{
        "label": i,
        "breed": names[i],
        "support": int(support[i]),
        "precision": round(float(precision[i]), 4),
        "recall": round(float(recall[i]), 4),
        "f1": round(float(f1[i]), 4),
    } for i in range(num_classes)]
    write_csv(output_dir / "per_class_metrics.csv",
              ["label", "breed", "support", "precision", "recall", "f1"], per_class)

    np.savetxt(output_dir / "confusion_matrix.csv", confusion.numpy(), fmt="%d", delimiter=",")

    mistakes = Counter()
    matrix = confusion.numpy()
    for true_index in range(num_classes):
        for predicted_index in range(num_classes):
            if true_index != predicted_index and matrix[true_index, predicted_index]:
                mistakes[(true_index, predicted_index)] = int(matrix[true_index, predicted_index])
    top = [{
        "true_breed": names[t],
        "predicted_breed": names[p],
        "times": n,
        "share_of_breed": round(n / max(int(support[t]), 1), 3),
    } for (t, p), n in mistakes.most_common(TOP_CONFUSIONS)]
    write_csv(output_dir / "top_confusions.csv",
              ["true_breed", "predicted_breed", "times", "share_of_breed"], top)

    # ----- figures -----
    figure_curves(run_dir, output_dir)
    figure_worst_confusion(confusion, names, f1, output_dir)
    figure_examples(rows, names, output_dir, True, "fig_correct.png",
                    "Correct predictions, most confident first")
    figure_examples(rows, names, output_dir, False, "fig_wrong.png",
                    "Mistakes, most confident first. Confident errors are the interesting ones")

    # ----- summary -----
    total = len(rows)
    correct_count = sum(r["correct"] for r in rows)
    low, high = wilson_interval(correct_count, total)
    top5 = sum(r["true_in_top5"] for r in rows) / total
    best_breeds = sorted(per_class, key=lambda r: -r["f1"])[:5]
    worst_breeds = sorted(per_class, key=lambda r: r["f1"])[:5]

    lines = [
        f"EVALUATION OF {args.run.upper()} ON THE {split.upper()} SET",
        "=" * 78,
        f"Weights                  epoch {checkpoint['epoch']} "
        f"(validation macro F1 {checkpoint['val_macro_f1']:.4f} when trained)",
        f"Trainable parameters     {count_parameters(model):,}",
        f"Photographs scored       {total}",
        f"Breeds                   {num_classes}, chance level "
        f"{100 / num_classes:.2f} per cent",
        "",
        "HEADLINE",
        f"  Accuracy               {100 * summary['accuracy']:.2f} per cent "
        f"(95 per cent interval {100 * low:.2f} to {100 * high:.2f})",
        f"  Top-5 accuracy         {100 * top5:.2f} per cent",
        f"  Macro F1               {summary['macro_f1']:.4f}",
        f"  Macro precision        {summary['macro_precision']:.4f}",
        f"  Macro recall           {summary['macro_recall']:.4f}",
        f"  Weighted F1            {summary['weighted_f1']:.4f}",
        "",
        "STRONGEST BREEDS BY F1",
    ]
    lines += [f"  {r['breed']:<34} F1 {r['f1']:.3f}  "
              f"({r['support']} photographs)" for r in best_breeds]
    lines += ["", "WEAKEST BREEDS BY F1"]
    lines += [f"  {r['breed']:<34} F1 {r['f1']:.3f}  "
              f"({r['support']} photographs)" for r in worst_breeds]
    lines += ["", f"MOST FREQUENT MISTAKES, top {min(8, len(top))}"]
    lines += [f"  {r['true_breed']:<30} called {r['predicted_breed']:<30} "
              f"{r['times']} times" for r in top[:8]]

    results = {"split": split, "photographs": total, "top5_accuracy": top5, **summary}

    if split == "test":
        leaked = leaked_keys("test")
        clean = subset_metrics(rows, num_classes, lambda r: r["photo"] not in leaked)
        results["clean_test"] = clean
        difference = 100 * (summary["accuracy"] - clean["accuracy"])
        lines += [
            "",
            "LEAKAGE CHECK, the measurement Stage 2 set up",
            f"  Full test set          {total} photographs, "
            f"{100 * summary['accuracy']:.2f} per cent",
            f"  Clean test set         {clean['photographs']} photographs, "
            f"{100 * clean['accuracy']:.2f} per cent",
            f"  Difference             {difference:+.2f} percentage points",
            "  The clean set leaves out the test photographs that have a confirmed",
            "  duplicate in training. A positive difference is the inflation the",
            "  duplicates caused. For this model the effect should be small, since it",
            "  never saw ImageNet, which is itself a result worth reporting.",
        ]

    summary_text = "\n".join(lines)
    (output_dir / "metrics_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(summary_text)
    print(f"\nWrote results to {output_dir}")
    if split == "val":
        print("\nThe test set has not been read. Run with --final once the model is settled.")


if __name__ == "__main__":
    main()
