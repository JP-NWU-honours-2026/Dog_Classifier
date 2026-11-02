"""
compare_models.py

One scorer for all three models, so the comparison in the report is produced by
a single piece of code rather than by three separate implementations.

Each person exports a predictions file from their own evaluation, and this
script computes every metric from it. Nobody has to adopt anyone else's model
code, and the numbers become directly comparable.

**It also checks that the three models were scored on the same photographs.**
If one model's test set differs, the comparison is not valid, and the report
would be wrong in a way that is hard to spot. That check runs automatically.

Only the Python standard library is used, so this runs anywhere with Python 3
and no installation.

-------------------------------------------------------------------------------
WHAT YOUR PREDICTIONS FILE NEEDS
-------------------------------------------------------------------------------
A CSV with one row per test photograph and these columns, in any order:

    photo            the photograph, for example n02085620-Chihuahua/n02085620_7.jpg
                     (a full path is fine, only the last two parts are used)
    true_breed       the correct breed
    predicted_breed  what the model said

Optional, used when present:

    confidence       the model's probability for its prediction, 0 to 1
    true_in_top5     True or False, whether the correct breed was in the top five

Column names may also be true_label and predicted_label, or true and predicted.
If the labels are numbers rather than names, pass --classes with a file that
maps them, or the numbers are used as names.

-------------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------------
    python compare_models.py baseline_cnn=path/to/predictions.csv \
                             efficientnetv2=path/to/jp.csv \
                             swin_t=path/to/sulaiman.csv \
                             --out results/comparison

Optional:
    --leaked  results/duplicate_check/leaked_photos.csv
              Also reports a clean score with those photographs excluded.
    --meta    a small JSON file giving each model's parameter count and
              training details, so they appear in the comparison table:
              {"swin_t": {"parameters": 27611634, "minutes": 19, "gpu": "RTX 4060",
                          "weights": "ImageNet pretrained"}}

Writes to the output folder:
    comparison.csv          one row per model, for pasting into the report
    comparison.md           the same table in markdown
    <model>_metrics.json    full metrics per model
    <model>_per_class.csv   precision, recall and F1 for all 120 breeds
    <model>_confusions.csv  the most frequent mistakes
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOP_CONFUSIONS = 20

PHOTO_FIELDS = ("photo", "image", "path", "file", "filename")
TRUE_FIELDS = ("true_breed", "true_label", "true", "actual", "label", "y_true")
PRED_FIELDS = ("predicted_breed", "predicted_label", "predicted", "prediction", "y_pred")
CONF_FIELDS = ("confidence", "probability", "score", "prob")
TOP5_FIELDS = ("true_in_top5", "in_top5", "top5_correct")


def pick(fieldnames, candidates):
    """Finds whichever of the accepted column names this file actually uses."""
    lowered = {name.lower().strip(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def photo_key(value):
    """
    Reduces any path to 'breed_folder/file_name', so files written on different
    computers still line up. Handles both slash styles.
    """
    parts = [p for p in re.split(r"[\\/]+", value.strip()) if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else value.strip()


def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def load_predictions(path, classes=None):
    """Reads one predictions file into a list of rows with consistent names."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header row.")

        photo_col = pick(reader.fieldnames, PHOTO_FIELDS)
        true_col = pick(reader.fieldnames, TRUE_FIELDS)
        pred_col = pick(reader.fieldnames, PRED_FIELDS)
        conf_col = pick(reader.fieldnames, CONF_FIELDS)
        top5_col = pick(reader.fieldnames, TOP5_FIELDS)

        missing = [label for label, col in
                   (("a photograph column", photo_col), ("a true label column", true_col),
                    ("a predicted label column", pred_col)) if col is None]
        if missing:
            raise SystemExit(
                f"{path} is missing {', '.join(missing)}.\n"
                f"Columns found: {', '.join(reader.fieldnames)}\n"
                f"Expected something like: photo, true_breed, predicted_breed")

        rows = []
        for row in reader:
            true_name = row[true_col].strip()
            pred_name = row[pred_col].strip()
            if classes:
                true_name = classes.get(true_name, true_name)
                pred_name = classes.get(pred_name, pred_name)
            rows.append({
                "photo": photo_key(row[photo_col]),
                "true": true_name,
                "predicted": pred_name,
                "confidence": float(row[conf_col]) if conf_col and row[conf_col] else None,
                "in_top5": truthy(row[top5_col]) if top5_col and row[top5_col] else None,
            })

    if not rows:
        raise SystemExit(f"{path} has no rows.")
    return rows


def wilson_interval(successes, trials, z=1.96):
    """A 95 per cent interval that behaves sensibly on small or lopsided counts."""
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def score(rows):
    """
    Accuracy and the macro and weighted averages, worked out from counts.

    Precision asks how many of the photographs called a breed really were it.
    Recall asks how many of that breed were found. F1 balances the two. Macro
    averaging weights all breeds equally, so a model cannot look good by doing
    well only on the breeds with the most photographs.
    """
    breeds = sorted({r["true"] for r in rows} | {r["predicted"] for r in rows})
    correct_of = Counter()
    predicted_of = Counter()
    actual_of = Counter()
    confusion = Counter()

    for r in rows:
        actual_of[r["true"]] += 1
        predicted_of[r["predicted"]] += 1
        if r["true"] == r["predicted"]:
            correct_of[r["true"]] += 1
        else:
            confusion[(r["true"], r["predicted"])] += 1

    per_class = []
    for breed in breeds:
        precision = correct_of[breed] / predicted_of[breed] if predicted_of[breed] else 0.0
        recall = correct_of[breed] / actual_of[breed] if actual_of[breed] else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class.append({
            "breed": breed,
            "support": actual_of[breed],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    total = len(rows)
    correct = sum(correct_of.values())
    scored = [c for c in per_class if c["support"] > 0]
    weight = {c["breed"]: c["support"] / total for c in per_class}
    low, high = wilson_interval(correct, total)

    graded = [r for r in rows if r["in_top5"] is not None]
    top5 = sum(r["in_top5"] for r in graded) / len(graded) if graded else None

    confident_errors = [r for r in rows
                        if r["confidence"] is not None
                        and r["true"] != r["predicted"] and r["confidence"] >= 0.9]

    return {
        "photographs": total,
        "breeds": len(scored),
        "accuracy": correct / total,
        "accuracy_interval_low": low,
        "accuracy_interval_high": high,
        "top5_accuracy": top5,
        "macro_precision": sum(c["precision"] for c in scored) / len(scored),
        "macro_recall": sum(c["recall"] for c in scored) / len(scored),
        "macro_f1": sum(c["f1"] for c in scored) / len(scored),
        "weighted_f1": sum(c["f1"] * weight[c["breed"]] for c in per_class),
        "errors": total - correct,
        "confident_errors": len(confident_errors) if any(
            r["confidence"] is not None for r in rows) else None,
    }, per_class, confusion


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_same_photographs(models):
    """
    The comparison is only valid if every model was scored on the same
    photographs. A model evaluated on a different or cleaned test set is not
    measuring the same thing, and that is easy to miss in a table of numbers.
    """
    sets = {name: {r["photo"] for r in rows} for name, rows in models.items()}
    sizes = {name: len(s) for name, s in sets.items()}
    lines = ["PHOTOGRAPHS SCORED", "-" * 78]
    for name, size in sizes.items():
        lines.append(f"  {name:<28} {size:>6}")

    shared = set.intersection(*sets.values()) if len(sets) > 1 else next(iter(sets.values()))
    identical = all(s == shared for s in sets.values())

    if identical:
        lines.append(f"\n  All models scored the same {len(shared)} photographs. "
                     f"The comparison is valid.")
    else:
        lines.append(f"\n  WARNING: the models were NOT scored on the same photographs.")
        lines.append(f"  Shared by all: {len(shared)}")
        for name, s in sets.items():
            extra, absent = len(s - shared), len(shared - s)
            if extra or absent:
                lines.append(f"    {name}: {extra} not in the others, "
                             f"{absent} of the shared set missing")
        lines.append("  Accuracies computed on different sets are not comparable.")
        lines.append("  Either rescore on a common set, or report them separately and")
        lines.append("  say in the report that the test sets differ.")
    return "\n".join(lines), identical, shared


def main():
    parser = argparse.ArgumentParser(
        description="Score several models from their prediction files, with one piece of code.")
    parser.add_argument("models", nargs="+", metavar="NAME=FILE",
                        help="for example swin_t=path/to/predictions.csv")
    parser.add_argument("--out", default="results/comparison", help="output folder")
    parser.add_argument("--leaked", help="leaked_photos.csv, to also report a clean score")
    parser.add_argument("--meta", help="JSON file of parameter counts and training details")
    parser.add_argument("--classes", help="JSON mapping label numbers to breed names")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = None
    if args.classes:
        raw = json.loads(Path(args.classes).read_text(encoding="utf-8"))
        classes = {str(k): (v["display_name"] if isinstance(v, dict) else v)
                   for k, v in (raw.get("classes", raw)).items()}

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8")) if args.meta else {}

    leaked = set()
    if args.leaked and Path(args.leaked).is_file():
        with open(args.leaked, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("pile", "test").strip() == "test":
                    leaked.add(photo_key(row["photo"]))

    models = {}
    for entry in args.models:
        if "=" not in entry:
            raise SystemExit(f"Expected NAME=FILE, got {entry!r}")
        name, path = entry.split("=", 1)
        models[name] = load_predictions(path, classes)
        print(f"  read {len(models[name]):>5} predictions for {name}")

    validity, identical, shared = check_same_photographs(models)
    print()
    print(validity)

    table = []
    for name, rows in models.items():
        summary, per_class, confusion = score(rows)
        info = meta.get(name, {})

        clean = None
        if leaked:
            kept = [r for r in rows if r["photo"] not in leaked]
            if len(kept) < len(rows):
                clean, _, _ = score(kept)

        write_csv(output_dir / f"{name}_per_class.csv",
                  ["breed", "support", "precision", "recall", "f1"],
                  sorted(per_class, key=lambda c: -c["f1"]))
        write_csv(output_dir / f"{name}_confusions.csv",
                  ["true_breed", "predicted_breed", "times"],
                  [{"true_breed": t, "predicted_breed": p, "times": n}
                   for (t, p), n in confusion.most_common(TOP_CONFUSIONS)])
        (output_dir / f"{name}_metrics.json").write_text(
            json.dumps({"model": name, "full": summary, "clean": clean, "meta": info},
                       indent=2), encoding="utf-8")

        table.append({
            "model": name,
            "weights": info.get("weights", ""),
            "parameters": info.get("parameters", ""),
            "photographs": summary["photographs"],
            "accuracy": round(100 * summary["accuracy"], 2),
            "interval": f"{100 * summary['accuracy_interval_low']:.2f} to "
                        f"{100 * summary['accuracy_interval_high']:.2f}",
            "top5": round(100 * summary["top5_accuracy"], 2) if summary["top5_accuracy"] else "",
            "macro_f1": round(summary["macro_f1"], 4),
            "macro_precision": round(summary["macro_precision"], 4),
            "macro_recall": round(summary["macro_recall"], 4),
            "clean_accuracy": round(100 * clean["accuracy"], 2) if clean else "",
            "clean_macro_f1": round(clean["macro_f1"], 4) if clean else "",
            "training_minutes": info.get("minutes", ""),
            "gpu": info.get("gpu", ""),
        })

    fields = list(table[0].keys())
    write_csv(output_dir / "comparison.csv", fields, table)

    header = ["Model", "Weights", "Parameters", "Photos", "Accuracy", "95% interval",
              "Top-5", "Macro F1", "Clean accuracy", "Training"]
    lines = ["# Model comparison", "",
             "Produced by compare_models.py, so every figure in this table is computed",
             "by the same code from each model's own predictions.", "",
             "| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in table:
        lines.append("| " + " | ".join(str(x) for x in [
            row["model"], row["weights"], f"{row['parameters']:,}" if row["parameters"] else "",
            row["photographs"], f"{row['accuracy']} %", row["interval"],
            f"{row['top5']} %" if row["top5"] else "",
            row["macro_f1"], f"{row['clean_accuracy']} %" if row["clean_accuracy"] else "",
            f"{row['training_minutes']} min, {row['gpu']}" if row["gpu"] else "",
        ]) + " |")
    lines += ["", "## Validity check", "", "```", validity, "```"]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("COMPARISON")
    print("-" * 78)
    print(f"{'model':<24}{'photos':>8}{'accuracy':>11}{'top-5':>9}{'macro F1':>10}{'clean':>9}")
    for row in table:
        print(f"{row['model']:<24}{row['photographs']:>8}{row['accuracy']:>10}%"
              f"{(str(row['top5']) + '%') if row['top5'] else '':>9}"
              f"{row['macro_f1']:>10}"
              f"{(str(row['clean_accuracy']) + '%') if row['clean_accuracy'] else '':>9}")
    print()
    print(f"Wrote {output_dir}")
    if not identical:
        print("\nRead the validity warning above before using this table.")


if __name__ == "__main__":
    main()
