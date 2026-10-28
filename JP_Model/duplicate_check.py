"""
duplicate_check.py

Stage 2 / Phase 1: Near-duplicate detection, data leakage audit, and candidate export.
Identifies duplicate photos and cross-breed label conflicts across all dataset splits.
Exports rich candidate JSON records for interactive visual review in the CanineVision Web App.

Writes to JP_Model/results/duplicate_check/:
    fingerprints.csv             Calculated hashes (dHash, pHash, SHA-256)
    confirmed_copies.csv         All confirmed duplicate pairs
    label_conflicts.csv          Duplicate pairs with conflicting breed labels
    leaked_photos.csv            Test/Val photos that have copies in training
    candidate_duplicates.json    Full visual audit schema for UI inspection and approval
    duplicate_check_summary.txt  Headline figures and statistics
"""

import csv
import hashlib
import io
import json
import random
import time
from collections import Counter
from pathlib import Path, PureWindowsPath

import imagehash
import numpy as np
from PIL import Image

JP_DIR = Path(__file__).resolve().parent
REPO_ROOT = JP_DIR.parent
IMAGES_DIR = REPO_ROOT / "archive" / "images" / "Images"
MANIFEST_PATH = JP_DIR / "split_manifest.json"
OUTPUT_DIR = JP_DIR / "results" / "duplicate_check"
FINGERPRINTS_PATH = OUTPUT_DIR / "fingerprints.csv"
LINDANI_COPIES_PATH = REPO_ROOT / "Lindani_Model" / "results" / "duplicate_check" / "confirmed_copies.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISTANCE = 10
CONFIRM_DHASH = 5
CONFIRM_PHASH = 10
PILES = ("train", "val", "test")

# Fast popcount for bit difference calculation
if hasattr(np, "bitwise_count"):
    def popcount(values):
        return np.bitwise_count(values)
else:
    _BITS_IN_BYTE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    def popcount(values):
        return _BITS_IN_BYTE[values.view(np.uint8)].reshape(-1, 8).sum(axis=1)


def local_key(entry):
    file_name = PureWindowsPath(entry["path"]).name
    return f"{entry['breed_folder']}/{file_name}"


def load_manifest_photos():
    """Loads all 20,580 photos from split_manifest.json with breed names and pile."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    classes = manifest.get("classes", {})

    photos = []
    for pile in PILES:
        for entry in manifest["splits"][pile]:
            key = local_key(entry)
            breed_folder = entry["breed_folder"]
            # Extract readable name
            parts = breed_folder.split("-", 1)
            display_name = parts[1].replace("_", " ").title() if len(parts) > 1 else breed_folder
            label_id = entry.get("label", -1)
            if str(label_id) in classes and "display_name" in classes[str(label_id)]:
                display_name = classes[str(label_id)]["display_name"]

            photos.append({
                "key": key,
                "breed_folder": breed_folder,
                "breed_name": display_name,
                "pile": pile,
                "label": label_id,
                "path": str(IMAGES_DIR / key),
                "url": f"/archive_images/{key}"
            })

    return sorted(photos, key=lambda ph: ph["key"]), manifest


def compute_or_load_fingerprints(photos):
    """Computes SHA-256, dHash, and pHash for all photos, caching to fingerprints.csv."""
    if FINGERPRINTS_PATH.exists():
        with open(FINGERPRINTS_PATH, newline="", encoding="utf-8") as f:
            saved = {row["key"]: row for row in csv.DictReader(f)}
        if set(saved.keys()) == {ph["key"] for ph in photos}:
            for ph in photos:
                row = saved[ph["key"]]
                ph.update(sha256=row["sha256"], dhash=row["dhash"], phash=row["phash"])
            print(f"[duplicate_check] Loaded saved fingerprints for {len(photos)} photos.")
            return

    print(f"[duplicate_check] Computing fingerprints for {len(photos)} photos...")
    start = time.time()
    for n, ph in enumerate(photos, 1):
        img_path = IMAGES_DIR / ph["key"]
        try:
            data = img_path.read_bytes()
            ph["sha256"] = hashlib.sha256(data).hexdigest()
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                ph["dhash"] = str(imagehash.dhash(img))
                ph["phash"] = str(imagehash.phash(img))
        except Exception as err:
            ph["sha256"] = ""
            ph["dhash"] = ""
            ph["phash"] = ""
            print(f"  Warning: could not read {ph['key']}: {err}")

        if n % 2500 == 0 or n == len(photos):
            print(f"  Fingerprinted {n}/{len(photos)} ({time.time() - start:.1f}s)")

    # Save to disk
    with open(FINGERPRINTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "breed_folder", "pile", "sha256", "dhash", "phash"])
        writer.writeheader()
        for ph in photos:
            writer.writerow({
                "key": ph["key"],
                "breed_folder": ph["breed_folder"],
                "pile": ph["pile"],
                "sha256": ph["sha256"],
                "dhash": ph["dhash"],
                "phash": ph["phash"]
            })
    print(f"[duplicate_check] Fingerprints saved to {FINGERPRINTS_PATH}")


def find_candidate_pairs(photos):
    """Vectorized pairwise comparison across all 20,580 photos within MAX_DISTANCE."""
    usable = [i for i, ph in enumerate(photos) if ph.get("dhash")]
    dhashes = np.array([int(photos[i]["dhash"], 16) for i in usable], dtype=np.uint64)
    pairs = []

    print(f"[duplicate_check] Running vectorized pairwise Hamming distance comparisons...")
    for row in range(len(usable) - 1):
        distances = popcount(np.bitwise_xor(dhashes[row + 1:], dhashes[row]))
        for hit in np.nonzero(distances <= MAX_DISTANCE)[0]:
            i, j = usable[row], usable[row + 1 + int(hit)]
            p_dist = bin(int(photos[i]["phash"], 16) ^ int(photos[j]["phash"], 16)).count("1")
            pairs.append((i, j, int(distances[hit]), p_dist))

    return pairs


def build_candidate_records(pairs, photos):
    """Builds structured candidate records for the visual review UI and CSVs."""
    photo_map = {ph["key"]: ph for ph in photos}
    records = []
    confirmed_copies = []
    label_conflicts = []
    leaked_photos = []

    for idx, (i, j, d_dist, p_dist) in enumerate(pairs, 1):
        a, b = photos[i], photos[j]
        exact_copy = (a["sha256"] and a["sha256"] == b["sha256"]) or (d_dist == 0 and p_dist == 0)
        confirmed_copy = exact_copy or (d_dist <= CONFIRM_DHASH and p_dist <= CONFIRM_PHASH)
        same_breed = (a["breed_folder"] == b["breed_folder"])
        crosses_piles = (a["pile"] != b["pile"])
        pile_pair = "-".join(sorted((a["pile"], b["pile"])))

        # Determine candidate category
        if not same_breed:
            category = "label_conflict"
            reason = f"Same image filed under 2 different breeds: '{a['breed_name']}' vs '{b['breed_name']}'"
            default_action = "remove_both"
        elif crosses_piles:
            category = "cross_split_leak"
            reason = f"Duplicate dog photo split across '{a['pile']}' and '{b['pile']}'"
            # Default: remove from val/test to prevent test contamination
            default_action = "remove_a" if a["pile"] in ("test", "val") and b["pile"] == "train" else "remove_b"
        else:
            category = "same_split_duplicate"
            reason = f"Duplicate image within the same '{a['pile']}' split"
            default_action = "remove_b"

        rec = {
            "pair_id": idx,
            "category": category,
            "reason": reason,
            "dhash_distance": d_dist,
            "phash_distance": p_dist,
            "exact_copy": bool(exact_copy),
            "confirmed_copy": bool(confirmed_copy),
            "same_breed": bool(same_breed),
            "crosses_piles": bool(crosses_piles),
            "pile_pair": pile_pair,
            "default_action": default_action,
            "status": "pending",  # pending | approved_remove_a | approved_remove_b | approved_remove_both | rejected_keep_both
            "photo_a": {
                "key": a["key"],
                "url": a["url"],
                "breed_folder": a["breed_folder"],
                "breed_name": a["breed_name"],
                "pile": a["pile"],
                "label": a["label"]
            },
            "photo_b": {
                "key": b["key"],
                "url": b["url"],
                "breed_folder": b["breed_folder"],
                "breed_name": b["breed_name"],
                "pile": b["pile"],
                "label": b["label"]
            }
        }
        records.append(rec)

        if confirmed_copy:
            confirmed_copies.append(rec)
            if not same_breed:
                label_conflicts.append(rec)
            if crosses_piles and "train" in (a["pile"], b["pile"]):
                leaked_photo = a if a["pile"] in ("test", "val") else b
                copy_photo = b if a["pile"] in ("test", "val") else a
                leaked_photos.append({
                    "photo": leaked_photo["key"],
                    "pile": leaked_photo["pile"],
                    "breed": leaked_photo["breed_name"],
                    "copy_in_train": copy_photo["key"],
                    "copy_breed": copy_photo["breed_name"],
                    "dhash_distance": d_dist,
                    "phash_distance": p_dist,
                    "exact_copy": exact_copy,
                    "same_breed": same_breed
                })

    return records, confirmed_copies, label_conflicts, leaked_photos


def run_audit():
    print("=" * 70)
    print("JP_Model Near-Duplicate and Data Leakage Audit Engine")
    print("=" * 70)

    photos, manifest = load_manifest_photos()
    print(f"[duplicate_check] Loaded {len(photos)} photos from {MANIFEST_PATH.name}")

    compute_or_load_fingerprints(photos)
    pairs = find_candidate_pairs(photos)
    print(f"[duplicate_check] Found {len(pairs)} candidate pairs within distance {MAX_DISTANCE}")

    records, confirmed_copies, label_conflicts, leaked_photos = build_candidate_records(pairs, photos)

    # 1. Save candidate JSON for UI visual review
    candidates_json_path = OUTPUT_DIR / "candidate_duplicates.json"
    with open(candidates_json_path, "w", encoding="utf-8") as f:
        json.dump(confirmed_copies, f, indent=2)
    print(f"[duplicate_check] Exported {len(confirmed_copies)} confirmed candidates to {candidates_json_path}")

    # 2. Save confirmed copies CSV
    confirmed_csv_path = OUTPUT_DIR / "confirmed_copies.csv"
    with open(confirmed_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["photo_a", "photo_b", "breed_a", "breed_b", "pile_a", "pile_b",
                         "dhash_dist", "phash_dist", "exact", "same_breed", "crosses_piles", "category"])
        for r in confirmed_copies:
            writer.writerow([
                r["photo_a"]["key"], r["photo_b"]["key"],
                r["photo_a"]["breed_name"], r["photo_b"]["breed_name"],
                r["photo_a"]["pile"], r["photo_b"]["pile"],
                r["dhash_distance"], r["phash_distance"],
                r["exact_copy"], r["same_breed"], r["crosses_piles"], r["category"]
            ])

    # 3. Save label conflicts CSV
    conflicts_csv_path = OUTPUT_DIR / "label_conflicts.csv"
    with open(conflicts_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["photo_a", "photo_b", "breed_a", "breed_b", "pile_a", "pile_b", "dhash_dist", "phash_dist"])
        for r in label_conflicts:
            writer.writerow([
                r["photo_a"]["key"], r["photo_b"]["key"],
                r["photo_a"]["breed_name"], r["photo_b"]["breed_name"],
                r["photo_a"]["pile"], r["photo_b"]["pile"],
                r["dhash_distance"], r["phash_distance"]
            ])

    # 4. Save leaked photos CSV
    leaked_csv_path = OUTPUT_DIR / "leaked_photos.csv"
    with open(leaked_csv_path, "w", newline="", encoding="utf-8") as f:
        if leaked_photos:
            writer = csv.DictWriter(f, fieldnames=list(leaked_photos[0].keys()))
            writer.writeheader()
            writer.writerows(leaked_photos)

    # 5. Write Summary Report
    summary_path = OUTPUT_DIR / "duplicate_check_summary.txt"
    summary_text = f"""JP_MODEL DUPLICATE & DATA LEAKAGE AUDIT SUMMARY
======================================================================
Total Photos Audited:          {len(photos)}
Total Candidate Pairs (<=10):   {len(pairs)}
Confirmed Duplicate Pairs:     {len(confirmed_copies)}
  - Cross-Breed Conflicts:     {len(label_conflicts)} (Same image, conflicting labels)
  - Cross-Split Leaks:         {len([c for c in confirmed_copies if c['crosses_piles']])}
  - Test Photos Leaked:        {len([l for l in leaked_photos if l['pile'] == 'test'])}
  - Validation Photos Leaked:  {len([l for l in leaked_photos if l['pile'] == 'val'])}

Interactive Visual Review File: {candidates_json_path.name}
Approved Removals File:         approved_removals.json (Generated after UI review)
Clean Manifest Target:          split_manifest_clean.json
"""
    summary_path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)


if __name__ == "__main__":
    run_audit()
