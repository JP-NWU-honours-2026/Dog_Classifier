"""
evaluate.py  (Sulaiman_Model, Swin Transformer)

Step 5: scores the trained Swin-T and produces everything the report needs.

Built from Lindani's Lindani_Model/evaluate.py, so both models are scored by the
same code and produce the same tables and figures, side by side. It is
standalone here (the colours, the tile helper and the confidence interval are
copied in), so it needs nothing from Lindani's folder except the leaked-photo
list that dataset.py already reads.

THE RULE: by default this scores the VALIDATION set, which you may do as often
as you like. It reads the TEST set only with --final, and refuses to score it a
second time unless told --overwrite. If you look at test scores, change
something and retrain, the test score stops being an honest measure of unseen
photos. So: validation while deciding, --final once the model is settled.

Usage, from the repository root:

    python Sulaiman_Model/evaluate.py              scores validation
    python Sulaiman_Model/evaluate.py --final      scores test, once

Writes to Sulaiman_Model/results/runs/<run>/eval_<val|test>/:

    metrics_summary.txt      headline figures, ready to quote
    metrics.json             the same numbers, machine-readable
    per_class_metrics.csv    precision, recall, F1 and support for all 120 breeds
    confusion_matrix.csv     the full 120 x 120 matrix (rows true, columns predicted)
    top_confusions.csv       the 20 most frequent mistakes
    predictions.csv          every photo: true breed, prediction, confidence, top-5 hit
    fig_curves.png           loss, accuracy and validation macro F1 per epoch
    fig_confusion_full.png   the whole 120 x 120 matrix as a heatmap
    fig_confusion_worst.png  a readable close-up of the 15 weakest breeds
    fig_correct.png          correct predictions, most confident first
    fig_wrong.png            mistakes, most confident first

With --final it also scores the CLEAN test set (the 34 leaked photos removed).

RESULTS (18 October 2026), for quick reference. Run swin_t, weights from epoch 15.
------------------------------------------------------------------------------
                      Validation (3,025)    Test (3,200)      Clean test (3,166)
  Accuracy            86.61 %               85.38 %           85.50 %
  95% interval        85.35 to 87.78        84.11 to 86.56
  Top-5 accuracy      98.88 %               98.56 %
  Macro F1            0.8644                0.8490            0.8500
  Macro precision     0.8685                0.8542
  Macro recall        0.8649                0.8502

  Test is about 1 point below validation, inside overlapping intervals: the
  model was not tuned to the validation set by accident.

  Weakest on test: Eskimo Dog 0.410, Collie 0.439, Miniature Poodle 0.444,
  Lhasa 0.528, Siberian Husky 0.557. Top mistakes are look-alike pairs:
  Eskimo Dog -> Siberian Husky (11), Miniature -> Toy Poodle (9),
  English Foxhound -> Walker Hound (8), Collie <-> Border Collie (8 + 7).

  LEAKAGE FINDING (checked against leaked_photos.csv):
    25 leaked test photos whose training copy has the SAME breed: 25 of 25 right.
     9 leaked test photos whose training copy has a DIFFERENT breed label:
       0 of 9 right, and in all 9 the model answered the training copy's label.
    So the model recalls photos it trained on. The two effects cancel out in
    the headline (clean is 0.13 points HIGHER than full), but the 9 show that
    some "mistakes" are really label errors in the dataset.
"""

import argparse
import csv
import json
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # draw straight to files, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from dataset import build_loaders, find_images_dir, leaked_keys, SULAIMAN_DIR
from model import build_model, count_parameters

RUNS_DIR = SULAIMAN_DIR / "results" / "runs"
EXAMPLE_TILES, EXAMPLE_COLS = 12, 4
WORST_BREEDS = 15
TOP_CONFUSIONS = 20
THUMB = 200

# Same colours and style as Lindani's figures, so the report looks like one piece.
DATA = "#2F6F94"
ACCENT = "#B4532A"
MUTED = "#6B7780"
INK = "#1A1A1A"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def wilson_interval(successes, trials, z=1.96):
    """95 per cent confidence interval for a proportion (Wilson method), copied from Lindani."""
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def square_tile(images_dir, key, pad=(247, 247, 247)):
    """Fits a whole photo into a square tile, padding rather than cropping."""
    tile = Image.new("RGB", (THUMB, THUMB), pad)
    with Image.open(images_dir / key) as img:
        small = ImageOps.contain(img.convert("RGB"), (THUMB, THUMB))
    tile.paste(small, ((THUMB - small.width) // 2, (THUMB - small.height) // 2))
    return tile


def load_model(run_dir, device):
    """Loads the best weights saved during training."""
    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    dropout = checkpoint.get("config", {}).get("dropout", 0.3)
    # pretrained=False: the architecture only. The trained weights are loaded
    # straight after, so there is no need to download ImageNet weights again.
    model = build_model(num_classes=len(class_names), dropout=dropout, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, class_names, checkpoint


def predict(model, loader, device, num_classes):
    """
    Runs every photo through the model once. Returns the confusion matrix and
    one row per photo. eval() mode switches dropout off, and no_grad() skips
    the bookkeeping needed for learning, because nothing is learned here.
    """
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    rows = []
    with torch.no_grad():
        for images, labels, keys in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                scores = model(images)
            probabilities = torch.softmax(scores.float(), dim=1).cpu()
            confidence, predictions = probabilities.max(dim=1)
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
    Accuracy, macro and weighted precision / recall / F1, plus per-breed arrays.

    Precision: of the photos called this breed, how many really were.
    Recall:    of the photos that really were this breed, how many were found.
    F1:        the balance of the two.
    Macro:     every breed counts equally. Weighted: by number of photos.
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
    """Same numbers for a subset of photos, e.g. the clean test set."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    kept = [row for row in rows if keep(row)]
    for row in kept:
        confusion[row["true_label"], row["predicted_label"]] += 1
    summary, *_ = metrics_from_confusion(confusion)
    summary["photographs"] = len(kept)
    summary["top5_accuracy"] = sum(r["true_in_top5"] for r in kept) / max(len(kept), 1)
    return summary


def figure_curves(run_dir, output_dir):
    """
    The learning history from training's metrics.csv: loss, accuracy, and
    validation macro F1, with the best epoch marked. The brief requires
    training and validation loss curves and a validation accuracy or F1 curve.
    """
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    if not history:
        return
    epochs = [int(r["epoch"]) for r in history]
    col = lambda name: [float(r[name]) for r in history]
    train_loss, val_loss = col("train_loss"), col("val_loss")
    train_acc, val_acc = col("train_accuracy"), col("val_accuracy")
    val_f1 = col("val_macro_f1")
    best = max(range(len(val_f1)), key=lambda i: val_f1[i])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.6))

    ax1.plot(epochs, train_loss, color=DATA, label="training")
    ax1.plot(epochs, val_loss, color=ACCENT, label="validation")
    ax1.set_title("Loss. The gap between the lines is overfitting")
    ax1.set_ylabel("loss")

    ax2.plot(epochs, [100 * a for a in train_acc], color=DATA, label="training")
    ax2.plot(epochs, [100 * a for a in val_acc], color=ACCENT, label="validation")
    ax2.set_title("Accuracy")
    ax2.set_ylabel("accuracy (per cent)")

    ax3.plot(epochs, val_f1, color=DATA)
    ax3.plot([epochs[best]], [val_f1[best]], "o", color=ACCENT)
    ax3.annotate(f"best epoch {epochs[best]}, {val_f1[best]:.3f}",
                 (epochs[best], val_f1[best]), textcoords="offset points",
                 xytext=(-10, -16), fontsize=8, color=ACCENT, ha="right")
    ax3.set_title("Validation macro F1, which early stopping watched")
    ax3.set_ylabel("validation macro F1")

    for ax in (ax1, ax2, ax3):
        ax.axvline(epochs[best], color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_xlabel("epoch")
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)

    fig.suptitle("Swin-T (pretrained, fine-tuned)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_curves.png")
    plt.close(fig)


def figure_full_confusion(confusion, output_dir, split):
    """
    The whole 120 x 120 matrix. Each row is scaled to sum to 1 (share of that
    breed's photos), so a strong diagonal means most photos land on their own
    breed and any bright off-diagonal dot is a systematic confusion.
    """
    matrix = confusion.numpy().astype(float)
    rows = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, rows, out=np.zeros_like(matrix), where=rows > 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("predicted breed (label 0 to 119)")
    ax.set_ylabel("true breed (label 0 to 119)")
    ax.set_title(f"Confusion matrix, all 120 breeds, {split} set, rows normalised")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share of the true breed")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_confusion_full.png")
    plt.close(fig)


def figure_worst_confusion(confusion, names, f1, output_dir):
    """Readable close-up: the confusion submatrix of the 15 weakest breeds by F1."""
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


def figure_examples(rows, names, images_dir, output_dir, correct, filename, title):
    """Photos it got right or wrong, with what it said and how sure it was."""
    chosen = [r for r in rows if r["correct"] == correct]
    if not chosen:
        return
    chosen = sorted(chosen, key=lambda r: -r["confidence"])[:EXAMPLE_TILES]
    rows_needed = -(-len(chosen) // EXAMPLE_COLS)

    fig, axes = plt.subplots(rows_needed, EXAMPLE_COLS, figsize=(13, 4.0 * rows_needed),
                             constrained_layout=True)
    for ax, row in zip(np.array(axes).ravel(), chosen):
        ax.imshow(square_tile(images_dir, row["photo"]))
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
    parser = argparse.ArgumentParser(description="Evaluate the trained Swin-T.")
    parser.add_argument("--run", default="swin_t", help="run folder under results/runs")
    parser.add_argument("--final", action="store_true",
                        help="score the TEST set. Only once, for the final model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
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
            print("The test set has already been scored for this run. Only use "
                  "--overwrite if this is a rerun of the same, unchanged model.")
        else:
            print("Use --overwrite to replace it.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, names, checkpoint = load_model(run_dir, device)
    num_classes = len(names)

    loaders, loader_names, _ = build_loaders(batch_size=args.batch_size,
                                             num_workers=args.num_workers)
    assert loader_names == names, "the split file and the checkpoint disagree about breeds"
    images_dir = find_images_dir()

    print("Scoring the TEST set. This is the one that counts.\n" if split == "test"
          else "Scoring the VALIDATION set. The test set stays sealed.\n")

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
        "label": i, "breed": names[i], "support": int(support[i]),
        "precision": round(float(precision[i]), 4),
        "recall": round(float(recall[i]), 4),
        "f1": round(float(f1[i]), 4),
    } for i in range(num_classes)]
    write_csv(output_dir / "per_class_metrics.csv",
              ["label", "breed", "support", "precision", "recall", "f1"], per_class)

    np.savetxt(output_dir / "confusion_matrix.csv", confusion.numpy(), fmt="%d", delimiter=",")

    mistakes = Counter()
    matrix = confusion.numpy()
    for t in range(num_classes):
        for p in range(num_classes):
            if t != p and matrix[t, p]:
                mistakes[(t, p)] = int(matrix[t, p])
    top = [{
        "true_breed": names[t], "predicted_breed": names[p], "times": n,
        "share_of_breed": round(n / max(int(support[t]), 1), 3),
    } for (t, p), n in mistakes.most_common(TOP_CONFUSIONS)]
    write_csv(output_dir / "top_confusions.csv",
              ["true_breed", "predicted_breed", "times", "share_of_breed"], top)

    # ----- figures -----
    figure_curves(run_dir, output_dir)
    figure_full_confusion(confusion, output_dir, split)
    figure_worst_confusion(confusion, names, f1, output_dir)
    figure_examples(rows, names, images_dir, output_dir, True, "fig_correct.png",
                    "Correct predictions, most confident first")
    figure_examples(rows, names, images_dir, output_dir, False, "fig_wrong.png",
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
        f"Parameters               {count_parameters(model):,}",
        f"Photographs scored       {total}",
        f"Breeds                   {num_classes}, chance level {100 / num_classes:.2f} per cent",
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
    lines += [f"  {r['breed']:<34} F1 {r['f1']:.3f}  ({r['support']} photographs)"
              for r in best_breeds]
    lines += ["", "WEAKEST BREEDS BY F1"]
    lines += [f"  {r['breed']:<34} F1 {r['f1']:.3f}  ({r['support']} photographs)"
              for r in worst_breeds]
    lines += ["", f"MOST FREQUENT MISTAKES, top {min(8, len(top))}"]
    lines += [f"  {r['true_breed']:<30} called {r['predicted_breed']:<30} {r['times']} times"
              for r in top[:8]]

    results = {"split": split, "photographs": total, "top5_accuracy": top5,
               "accuracy_95_interval": [low, high], **summary}

    if split == "test":
        leaked = leaked_keys("test")
        clean = subset_metrics(rows, num_classes, lambda r: r["photo"] not in leaked)
        results["clean_test"] = clean
        difference = 100 * (summary["accuracy"] - clean["accuracy"])
        lines += [
            "",
            "LEAKAGE CHECK",
            f"  Full test set          {total} photographs, "
            f"{100 * summary['accuracy']:.2f} per cent, macro F1 {summary['macro_f1']:.4f}",
            f"  Clean test set         {clean['photographs']} photographs, "
            f"{100 * clean['accuracy']:.2f} per cent, macro F1 {clean['macro_f1']:.4f}",
            f"  Difference             {difference:+.2f} percentage points",
            "  The clean set leaves out test photos with a confirmed near-copy in",
            "  training (Lindani's duplicate check). A positive difference is the",
            "  inflation those copies caused.",
            "  Separate, larger caveat: Swin-T's ImageNet pretraining very likely",
            "  included these exact photos, since Stanford Dogs was built from",
            "  ImageNet. Removing 34 duplicates does not undo that.",
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
