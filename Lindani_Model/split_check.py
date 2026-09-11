"""
split_check.py

Stage 2, script 1: audit the group's shared train/validation/test split.

Reads the shared split file (split_manifest.json) without changing it, and answers:
  1. Does every photo named in the split exist on this machine?
  2. Does rerunning JP's splitting logic with seed 42 give the identical split?
  3. Does any photo appear in more than one pile?
  4. Do the labels agree with the breed folders?
  5. Does every breed appear in all three piles, and in what numbers?

Run from the repository root (the Dog_Classifier folder), with the virtual
environment active:
    python Lindani_Model/split_check.py

Writes to Lindani_Model/results/split_check/:
    split_counts_per_breed.csv   photo counts and percentages per breed, per pile
    split_check_summary.txt      the pass or fail result of every check

Uses only the Python standard library, so nothing needs installing.
"""

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path, PureWindowsPath

# ---------------------------------------------------------------------------
# Locations. Everything is worked out relative to this file, so the script
# runs the same on this laptop, on Kaggle, or on a teammate's computer.
# ---------------------------------------------------------------------------
LINDANI_DIR = Path(__file__).resolve().parent
REPO_ROOT = LINDANI_DIR.parent
IMAGES_DIR = REPO_ROOT / "archive" / "images" / "Images"
MANIFEST_PATH = LINDANI_DIR / "split_manifest.json"
OUTPUT_DIR = LINDANI_DIR / "results" / "split_check"

# The settings JP_Model/dataset.py used to make the split.
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PILES = ("train", "val", "test")


def local_key(entry):
    """
    Turns one entry from the split file into a short name that works on any
    machine: 'breed_folder/file_name'.

    JP's paths start with his own computer's address (C:\\Users\\jpret\\...),
    so only the file name at the end is kept. PureWindowsPath is used because
    his paths use Windows backslashes, and this must still work on Kaggle,
    which runs Linux.
    """
    file_name = PureWindowsPath(entry["path"]).name
    return f"{entry['breed_folder']}/{file_name}"


def reproduce_split():
    """
    Reruns the splitting steps from JP_Model/dataset.py (create_or_load_splits)
    on the photos in this repository, and returns the result as
    'breed_folder/file_name' names.

    The steps are copied exactly: sort the breed folders, set the seed once,
    then for each breed sort its photos, shuffle them, and cut the shuffled
    list at 70 per cent and 85 per cent. JP sorted full paths and this sorts
    file names, which gives the same order because each breed's photos all sit
    in one folder.
    """
    breed_folders = sorted(d.name for d in IMAGES_DIR.iterdir() if d.is_dir())
    random.seed(SEED)
    result = {pile: [] for pile in PILES}

    for folder in breed_folders:
        files = sorted(
            p.name for p in (IMAGES_DIR / folder).iterdir()
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        random.shuffle(files)

        n = len(files)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        result["train"] += [f"{folder}/{f}" for f in files[:train_end]]
        result["val"] += [f"{folder}/{f}" for f in files[train_end:val_end]]
        result["test"] += [f"{folder}/{f}" for f in files[val_end:]]

    return breed_folders, result


def pct(part, whole):
    return round(100 * part / whole, 1) if whole else 0.0


def main():
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    splits = manifest["splits"]
    classes = manifest["classes"]  # {"0": {"folder": ..., "display_name": ...}, ...}

    keys = {pile: [local_key(e) for e in splits[pile]] for pile in PILES}
    total_listed = sum(len(keys[p]) for p in PILES)
    checks = []  # (description, passed, detail)

    # Check 1: every photo named in the split exists on this machine.
    missing = [k for p in PILES for k in keys[p] if not (IMAGES_DIR / k).is_file()]
    checks.append((
        "1. Every photo in the split exists on this machine",
        not missing,
        f"{total_listed} listed, {len(missing)} missing",
    ))

    # Check 2: rerunning the split gives the identical result, in the same order.
    breed_folders, reproduced = reproduce_split()
    identical = all(reproduced[p] == keys[p] for p in PILES)
    checks.append((
        "2. Rerunning with seed 42 gives the identical split",
        identical,
        ", ".join(f"{p}: {len(reproduced[p])} rerun vs {len(keys[p])} in file" for p in PILES),
    ))

    # Check 3: no photo in two piles, and no photo listed twice within a pile.
    sets = {p: set(keys[p]) for p in PILES}
    shared = {
        f"{a} and {b}": len(sets[a] & sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    repeated = {p: len(keys[p]) - len(sets[p]) for p in PILES}
    checks.append((
        "3. No photo appears in more than one pile",
        not any(shared.values()) and not any(repeated.values()),
        "shared: " + ", ".join(f"{k} {v}" for k, v in shared.items())
        + "; listed twice: " + ", ".join(f"{k} {v}" for k, v in repeated.items()),
    ))

    # Check 4: label numbers follow the sorted breed folders, and every entry's
    # label and folder agree with each other and with its original path.
    expected_label = {folder: i for i, folder in enumerate(breed_folders)}
    class_list_ok = {c["folder"]: int(i) for i, c in classes.items()} == expected_label
    mislabelled = [
        e for p in PILES for e in splits[p]
        if expected_label.get(e["breed_folder"]) != e["label"]
        or PureWindowsPath(e["path"]).parent.name != e["breed_folder"]
    ]
    checks.append((
        "4. Labels agree with the breed folders",
        class_list_ok and not mislabelled,
        f"{len(classes)} classes in file, class list matches folders: {class_list_ok}, "
        f"mislabelled entries: {len(mislabelled)}",
    ))

    # Check 5: every breed appears in all three piles. Also builds the table.
    counts = {p: Counter(k.split("/")[0] for k in keys[p]) for p in PILES}
    display_name = {c["folder"]: c["display_name"] for c in classes.values()}
    rows = []
    for folder in breed_folders:
        n = {p: counts[p][folder] for p in PILES}
        total = sum(n.values())
        rows.append({
            "label": expected_label[folder],
            "breed_folder": folder,
            "breed_name": display_name.get(folder, folder),
            "total": total,
            "train": n["train"],
            "val": n["val"],
            "test": n["test"],
            "train_pct": pct(n["train"], total),
            "val_pct": pct(n["val"], total),
            "test_pct": pct(n["test"], total),
        })
    absent = [r["breed_folder"] for r in rows if min(r["train"], r["val"], r["test"]) == 0]
    checks.append((
        "5. Every breed appears in all three piles",
        not absent,
        f"{len(rows)} breeds, {len(absent)} missing from at least one pile",
    ))

    # ----- write the per-breed table -----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grand = {p: len(keys[p]) for p in PILES}
    rows.append({
        "label": "", "breed_folder": "TOTAL", "breed_name": "All breeds",
        "total": total_listed, **grand,
        "train_pct": pct(grand["train"], total_listed),
        "val_pct": pct(grand["val"], total_listed),
        "test_pct": pct(grand["test"], total_listed),
    })
    with open(OUTPUT_DIR / "split_counts_per_breed.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ----- write and print the summary -----
    breeds = rows[:-1]
    smallest = min(breeds, key=lambda r: r["total"])
    largest = max(breeds, key=lambda r: r["total"])
    lines = [
        "SHARED SPLIT AUDIT",
        "=" * 64,
        f"Split file audited   {MANIFEST_PATH.name}",
        f"SHA-256 of that file {hashlib.sha256(manifest_bytes).hexdigest()}",
        f"Seed in the file     {manifest['metadata'].get('seed')}",
        "",
        f"Photos               {total_listed}",
        f"Train                {grand['train']} ({pct(grand['train'], total_listed)} per cent)",
        f"Validation           {grand['val']} ({pct(grand['val'], total_listed)} per cent)",
        f"Test                 {grand['test']} ({pct(grand['test'], total_listed)} per cent)",
        "",
        f"Breeds               {len(breeds)}",
        f"Smallest breed       {smallest['breed_folder']} ({smallest['total']} photos)",
        f"Largest breed        {largest['breed_folder']} ({largest['total']} photos)",
        f"Train share range    {min(r['train_pct'] for r in breeds)} to "
        f"{max(r['train_pct'] for r in breeds)} per cent per breed",
        f"Validation range     {min(r['val_pct'] for r in breeds)} to "
        f"{max(r['val_pct'] for r in breeds)} per cent per breed",
        f"Test share range     {min(r['test_pct'] for r in breeds)} to "
        f"{max(r['test_pct'] for r in breeds)} per cent per breed",
        "",
        "CHECKS",
        "-" * 64,
    ]
    for description, passed, detail in checks:
        lines.append(f"[{'PASS' if passed else 'FAIL'}] {description}")
        lines.append(f"       {detail}")
    lines += [
        "-" * 64,
        "Not checked here: near-duplicate photos across piles. See script 2.",
    ]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "split_check_summary.txt").write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"\nWrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
