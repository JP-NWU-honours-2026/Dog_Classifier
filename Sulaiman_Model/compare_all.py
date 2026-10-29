"""
compare_all.py - the three-model comparison, scored by one piece of code.

Runs Lindani's shared scorer (Lindani_Model/compare_models.py) on each model's
per-photo predictions, so every number in the comparison table is computed the
same way, and the scorer checks that all models were scored on the same
photographs. Also draws the validation curves of the two pretrained models.

Two comparisons, because the split files define different tasks:

  original    120 classes, 3,200 test photos (split_manifest.json)
              baseline CNN (Lindani), EfficientNetV2-S (JP), Swin-T (Sulaiman)
              plus a "clean" score without the 36 test photos JP's cleaning
              removed (the same definition as split_manifest_clean.json)
  simplified  114 classes, 3,164 test photos (split_manifest_simplified.json)
              EfficientNetV2-S and Swin-T (the baseline was not trained on it)

Predictions files it reads (missing ones are skipped and reported):

  Lindani_Model/results/runs/iter2_deeper/eval_test/predictions.csv
  JP_Model/results/runs/unclean_transfer/eval_test/predictions.csv
  JP_Model/results/runs/simplified_transfer/eval_test/predictions.csv
  Sulaiman_Model/results/runs/unclean_transfer/eval_test/predictions.csv
  Sulaiman_Model/results/runs/simplified_transfer/eval_test/predictions.csv

JP's evaluate.py does not write predictions.csv yet; Sulaiman_Model/evaluate.py
does (the same few lines can be added to JP's). Nothing outside
Sulaiman_Model/ is changed: other folders are only read.

Run from the repository root:
    python Sulaiman_Model/compare_all.py

Writes to Sulaiman_Model/results/comparison/:
    original/    comparison.csv, comparison.md, <model>_metrics.json,
                 <model>_per_class.csv, <model>_confusions.csv
    simplified/  the same for the simplified split
    removed_test_photos.csv   the 36 photos left out of the clean score
    meta.json                 parameters, training time and GPU per model
    validation_curves.png     validation accuracy and loss per epoch
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "results" / "comparison"
SCORER = REPO / "Lindani_Model" / "compare_models.py"

RUNS = {
    # name: (predictions.csv, run folder with run_config.json / metrics.csv, member)
    "original": {
        "baseline_cnn": (REPO / "Lindani_Model/results/runs/iter2_deeper/eval_test/predictions.csv",
                         REPO / "Lindani_Model/results/runs/iter2_deeper", "Lindani"),
        "efficientnetv2_s": (REPO / "JP_Model/results/runs/unclean_transfer/eval_test/predictions.csv",
                             REPO / "JP_Model/results/runs/unclean_transfer", "JP"),
        "swin_t": (HERE / "results/runs/unclean_transfer/eval_test/predictions.csv",
                   HERE / "results/runs/unclean_transfer", "Sulaiman"),
    },
    "simplified": {
        "efficientnetv2_s": (REPO / "JP_Model/results/runs/simplified_transfer/eval_test/predictions.csv",
                             REPO / "JP_Model/results/runs/simplified_transfer", "JP"),
        "swin_t": (HERE / "results/runs/simplified_transfer/eval_test/predictions.csv",
                   HERE / "results/runs/simplified_transfer", "Sulaiman"),
    },
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def photo_key(path):
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:])


def removed_test_photos():
    """Test photos in split_manifest.json that JP's cleaning removed (split_manifest_clean.json)."""
    full = load_json(REPO / "JP_Model/split_manifest.json")["splits"]["test"]
    clean = load_json(REPO / "JP_Model/split_manifest_clean.json")["splits"]["test"]
    kept = {photo_key(e["path"]) for e in clean}
    removed = sorted(photo_key(e["path"]) for e in full if photo_key(e["path"]) not in kept)
    path = OUT / "removed_test_photos.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["photo", "pile"])
        writer.writerows([p, "test"] for p in removed)
    return path, len(removed)


def run_meta(run_dir):
    """Parameters, training minutes, GPU and weights from a run's run_config.json."""
    cfg = load_json(run_dir / "run_config.json")
    if "total_parameters" in cfg:   # JP's pipeline (JP and Sulaiman)
        return {"parameters": cfg["total_parameters"],
                "minutes": round(cfg.get("duration_minutes", 0), 1),
                "gpu": cfg.get("gpu_name", ""),
                "weights": "ImageNet-1K pretrained"}
    return {"parameters": cfg.get("trainable_parameters", ""),   # Lindani's pipeline
            "minutes": round(cfg.get("total_minutes", 0), 1),
            "gpu": cfg.get("hardware", {}).get("gpu", "") or cfg.get("gpu", ""),
            "weights": "none (from scratch)"}


def gpu_from_lindani(run_dir):
    """Lindani's run_config keeps the GPU name in a nested block; find it wherever it is."""
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "gpu" and isinstance(v, str):
                    return v
                found = walk(v)
                if found:
                    return found
        return None
    return walk(load_json(run_dir / "run_config.json")) or ""


def plot_curves():
    """Validation accuracy and loss per epoch for the pretrained runs (JP's metrics.csv format)."""
    series = [("Swin-T, original", HERE / "results/runs/unclean_transfer", "#2F6F94", "o-"),
              ("Swin-T, simplified", HERE / "results/runs/simplified_transfer", "#2F6F94", "s--"),
              ("EfficientNetV2-S, original", REPO / "JP_Model/results/runs/unclean_transfer", "#B4532A", "o-"),
              ("EfficientNetV2-S, simplified", REPO / "JP_Model/results/runs/simplified_transfer", "#B4532A", "s--")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    for label, run_dir, colour, style in series:
        path = run_dir / "metrics.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epochs = [int(r["epoch"]) for r in rows]
        ax1.plot(epochs, [float(r["val_accuracy"]) for r in rows], style, color=colour, label=label)
        ax2.plot(epochs, [float(r["val_loss"]) for r in rows], style, color=colour, label=label)
    for ax, title, ylabel in ((ax1, "Validation accuracy per epoch", "validation accuracy (%)"),
                              (ax2, "Validation loss per epoch", "validation loss")):
        ax.axvline(2.5, color="grey", linestyle=":", linewidth=1)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("epoch (1-2 head only, 3-5 fine-tuning)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    fig.suptitle("Same pipeline, recipe and split files: Swin-T vs EfficientNetV2-S", fontsize=12)
    fig.tight_layout()
    path = OUT / "validation_curves.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main():
    if not SCORER.exists():
        sys.exit(f"{SCORER} not found. It is on Lindani's branch: merge it first "
                 "(git fetch, then git merge origin/Lindani-branch). It only adds files "
                 "inside Lindani_Model/.")
    OUT.mkdir(parents=True, exist_ok=True)

    removed_csv, n_removed = removed_test_photos()
    print(f"Clean score leaves out {n_removed} test photos (JP's cleaning): {removed_csv}")

    meta, skipped = {}, []
    for comparison, models in RUNS.items():
        args = []
        for name, (preds, run_dir, member) in models.items():
            if not preds.exists():
                skipped.append(f"{comparison}/{name} ({member}): no {preds.relative_to(REPO)}")
                continue
            args.append(f"{name}={preds}")
            info = run_meta(run_dir)
            if not info["gpu"]:
                info["gpu"] = gpu_from_lindani(run_dir)
            meta.setdefault(comparison, {})[name] = info
        if not args:
            continue
        meta_path = OUT / f"meta_{comparison}.json"
        meta_path.write_text(json.dumps(meta[comparison], indent=2), encoding="utf-8")
        command = [sys.executable, str(SCORER), *args, "--out", str(OUT / comparison),
                   "--meta", str(meta_path)]
        if comparison == "original":
            command += ["--leaked", str(removed_csv)]
        print(f"\n===== {comparison.upper()} SPLIT =====")
        subprocess.run(command, check=True)

    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for p in OUT.glob("meta_*.json"):
        p.unlink()
    print(f"\nCurves: {plot_curves()}")

    if skipped:
        print("\nSkipped (no predictions file yet):")
        for s in skipped:
            print(f"  {s}")
        print("JP's evaluate.py needs the predictions.csv export that Sulaiman_Model/evaluate.py has.")


if __name__ == "__main__":
    main()
