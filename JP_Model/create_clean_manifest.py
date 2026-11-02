"""
create_clean_manifest.py

Phase 3: Creates split_manifest_clean.json by filtering out user-approved duplicates
and conflicting labels across all dataset splits.
"""

import json
from collections import Counter
from pathlib import Path, PureWindowsPath

JP_DIR = Path(__file__).resolve().parent
REPO_ROOT = JP_DIR.parent
MANIFEST_PATH = JP_DIR / "split_manifest.json"
CLEAN_MANIFEST_PATH = JP_DIR / "split_manifest_clean.json"
REMOVALS_PATH = JP_DIR / "results" / "duplicate_check" / "approved_removals.json"
CANDIDATES_PATH = JP_DIR / "results" / "duplicate_check" / "candidate_duplicates.json"
SUMMARY_PATH = JP_DIR / "results" / "duplicate_check" / "clean_manifest_summary.json"


def local_key(path_str, breed_folder):
    file_name = PureWindowsPath(path_str).name
    return f"{breed_folder}/{file_name}"


def get_approved_removal_keys():
    """Gathers set of all photo keys approved for removal by the user."""
    removed_keys = set()

    # 1. From approved_removals.json
    if REMOVALS_PATH.exists():
        try:
            removals = json.loads(REMOVALS_PATH.read_text(encoding="utf-8"))
            if isinstance(removals, list):
                for item in removals:
                    if isinstance(item, str):
                        removed_keys.add(item)
                    elif isinstance(item, dict) and "key" in item:
                        removed_keys.add(item["key"])
            elif isinstance(removals, dict):
                removed_keys.update(removals.keys())
        except Exception as e:
            print(f"Warning reading {REMOVALS_PATH}: {e}")

    # 2. Also check candidate_duplicates.json for any decisions made in the UI
    if CANDIDATES_PATH.exists():
        try:
            candidates = json.loads(CANDIDATES_PATH.read_text(encoding="utf-8"))
            for c in candidates:
                status = c.get("status", "")
                if status == "approved_remove_a" or status == "remove_a":
                    removed_keys.add(c["photo_a"]["key"])
                elif status == "approved_remove_b" or status == "remove_b":
                    removed_keys.add(c["photo_b"]["key"])
                elif status == "approved_remove_both" or status == "remove_both":
                    removed_keys.add(c["photo_a"]["key"])
                    removed_keys.add(c["photo_b"]["key"])
        except Exception as e:
            print(f"Warning reading {CANDIDATES_PATH}: {e}")

    return removed_keys


def build_clean_manifest():
    print("=" * 70)
    print("JP_Model Clean Split Manifest Generator")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Source manifest not found: {MANIFEST_PATH}")

    raw_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    removal_keys = get_approved_removal_keys()
    print(f"[clean_manifest] Found {len(removal_keys)} photos approved for removal.")

    clean_manifest = {
        "metadata": {
            **raw_manifest.get("metadata", {}),
            "manifest_type": "clean_deduplicated",
            "source_manifest": "split_manifest.json",
            "total_removals": len(removal_keys)
        },
        "classes": raw_manifest["classes"],
        "splits": {
            "train": [],
            "val": [],
            "test": []
        }
    }

    removed_stats = {"train": 0, "val": 0, "test": 0}
    retained_stats = {"train": 0, "val": 0, "test": 0}

    for pile in ("train", "val", "test"):
        for entry in raw_manifest["splits"][pile]:
            key = local_key(entry["path"], entry["breed_folder"])
            if key in removal_keys:
                removed_stats[pile] += 1
            else:
                retained_stats[pile] += 1
                clean_manifest["splits"][pile].append(entry)

    total_orig = sum(len(raw_manifest["splits"][p]) for p in ("train", "val", "test"))
    total_clean = sum(len(clean_manifest["splits"][p]) for p in ("train", "val", "test"))

    # Verify all 120 breeds remain represented across all splits
    for pile in ("train", "val", "test"):
        breed_set = {e["breed_folder"] for e in clean_manifest["splits"][pile]}
        if len(breed_set) < 120:
            print(f"Warning: {pile} split has only {len(breed_set)} breeds represented out of 120!")

    CLEAN_MANIFEST_PATH.write_text(json.dumps(clean_manifest, indent=2), encoding="utf-8")
    print(f"[clean_manifest] Saved clean split manifest to {CLEAN_MANIFEST_PATH}")

    summary = {
        "original_total": total_orig,
        "clean_total": total_clean,
        "total_removed": len(removal_keys),
        "removed_per_split": removed_stats,
        "retained_per_split": retained_stats,
        "classes_count": len(clean_manifest["classes"])
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(f"  Original Images: {total_orig}")
    print(f"  Clean Images:    {total_clean}")
    print(f"  Removed:         {len(removal_keys)}")
    print(f"    Train removed: {removed_stats['train']} (Retained: {retained_stats['train']})")
    print(f"    Val removed:   {removed_stats['val']} (Retained: {retained_stats['val']})")
    print(f"    Test removed:  {removed_stats['test']} (Retained: {retained_stats['test']})")

    return summary


if __name__ == "__main__":
    build_clean_manifest()
