"""
duplicate_check.py

Stage 2, script 2: look for near-duplicate photos, and measure whether copies of
the same picture have ended up in different piles of the shared split.

Section 7.1 of the project brief requires that near-identical images are not
divided across training and test. This script measures how far the shared split
meets that. It only reads the photos and the split file. It never moves, deletes
or edits anything.

How it works, in short:
  1. Every photo gets two fingerprints, each made of 64 yes/no answers about its
     pattern of light and dark: a difference hash (dHash, the main one) and a
     perceptual hash (pHash, a cross-check that works differently). It also gets
     a SHA-256 of the file bytes, to catch exact copies.
  2. Every photo's dHash is compared with every other photo's. The distance
     between two photos is how many of the 64 answers differ. 0 means they look
     the same, and small numbers mean they are probably copies.
  3. Every pair within distance 10 is recorded, with each photo's breed and pile.
  4. The results are summarised at several cut-offs, so the cut-off can be chosen
     from evidence rather than guessed.

Run from the repository root (the Dog_Classifier folder), with the virtual
environment active:
    python Lindani_Model/duplicate_check.py

The first run fingerprints all 20,580 photos, roughly 10 minutes on the laptop.
The fingerprints are saved, so later runs take seconds. Delete fingerprints.csv
to force a fresh pass.

Writes to Lindani_Model/results/duplicate_check/:
    fingerprints.csv             the saved fingerprints, one row per photo
    near_duplicate_pairs.csv     every pair within distance 10, with breeds and piles
    threshold_sweep.csv          what each cut-off flags, and how far the two methods agree
    duplicate_check_summary.txt  the headline figures in plain text
    pairs_distance_*.png         sample pairs at each distance, to check by eye

Needs: pip install pillow imagehash
"""

import csv
import hashlib
import io
import json
import random
import time
from collections import Counter

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Shared with script 1, so both scripts read the split in exactly the same way.
from split_check import IMAGES_DIR, LINDANI_DIR, MANIFEST_PATH, PILES, local_key

OUTPUT_DIR = LINDANI_DIR / "results" / "duplicate_check"
FINGERPRINTS_PATH = OUTPUT_DIR / "fingerprints.csv"

MAX_DISTANCE = 10                     # pairs further apart than this are ignored
SWEEP = (0, 2, 4, 5, 6, 8, 10)        # the cut-offs reported in the table

# The chosen rule for calling a pair a copy: both fingerprint methods must agree.
# Chosen from the evidence in threshold_sweep.csv and the sample sheets. At a
# dHash distance of 5 the sheets still show a few pairs that are plainly
# different photographs, and those all carry a pHash distance of 20 or more, so
# requiring both methods to agree removes them.
CONFIRM_DHASH = 5
CONFIRM_PHASH = 10
SHEET_BANDS = ((0, 0), (1, 2), (3, 4), (5, 5), (6, 6), (7, 8), (9, 10))
PAIRS_PER_SHEET = 12
SEED = 42                             # fixes which pairs are drawn on the sheets

PAIR_FIELDS = [
    "photo_a", "photo_b", "breed_a", "breed_b", "pile_a", "pile_b",
    "dhash_distance", "phash_distance", "confirmed_copy", "exact_copy",
    "same_breed", "crosses_piles", "pile_pair",
]

# One row per held-out photo that has a confirmed copy in the training pile.
LEAK_FIELDS = [
    "photo", "pile", "breed", "copy_in_train", "copy_breed",
    "dhash_distance", "phash_distance", "exact_copy", "same_breed",
]


# Counting the differing bits between two fingerprints. numpy 2 has this built
# in; the fallback counts the bits byte by byte using a lookup table.
if hasattr(np, "bitwise_count"):
    def popcount(values):
        return np.bitwise_count(values)
else:
    _BITS_IN_BYTE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount(values):
        return _BITS_IN_BYTE[values.view(np.uint8)].reshape(-1, 8).sum(axis=1)


def load_split():
    """One entry per photo: its short name, breed folder and pile, sorted by name."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    photos = [
        {"key": local_key(entry), "breed": entry["breed_folder"], "pile": pile}
        for pile in PILES
        for entry in manifest["splits"][pile]
    ]
    return sorted(photos, key=lambda ph: ph["key"])


def fingerprint(photos):
    """
    Adds sha256, dhash and phash to every photo. Reuses fingerprints.csv when it
    covers exactly the same photos. Opens one photo at a time, so memory use
    stays small on an 8GB laptop.
    """
    if FINGERPRINTS_PATH.exists():
        with open(FINGERPRINTS_PATH, newline="", encoding="utf-8") as f:
            saved = {row["key"]: row for row in csv.DictReader(f)}
        if set(saved) == {ph["key"] for ph in photos}:
            for ph in photos:
                row = saved[ph["key"]]
                ph.update(sha256=row["sha256"], dhash=row["dhash"], phash=row["phash"])
            print(f"Loaded saved fingerprints for {len(photos)} photos.")
            return
        print("Saved fingerprints do not match the split, so fingerprinting again.")

    start = time.time()
    for n, ph in enumerate(photos, 1):
        data = (IMAGES_DIR / ph["key"]).read_bytes()
        ph["sha256"] = hashlib.sha256(data).hexdigest()
        try:
            with Image.open(io.BytesIO(data)) as img:
                img.load()
                ph["dhash"] = str(imagehash.dhash(img))
                ph["phash"] = str(imagehash.phash(img))
        except Exception as error:  # a photo that will not open is reported, not fatal
            ph["dhash"] = ph["phash"] = ""
            print(f"  could not read {ph['key']}: {error}")
        if n % 1000 == 0 or n == len(photos):
            print(f"  fingerprinted {n}/{len(photos)} ({time.time() - start:.0f} s)")

    with open(FINGERPRINTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "breed", "pile", "sha256", "dhash", "phash"])
        writer.writeheader()
        writer.writerows(photos)


def find_pairs(photos):
    """
    Compares every photo's dHash with every later photo's, and returns each pair
    within MAX_DISTANCE as (i, j, dhash distance, phash distance).

    Each photo is compared against all the photos after it in one numpy
    operation, rather than one pair at a time, which is what makes 212 million
    comparisons finish in under a minute.
    """
    usable = [i for i, ph in enumerate(photos) if ph["dhash"]]
    dhashes = np.array([int(photos[i]["dhash"], 16) for i in usable], dtype=np.uint64)
    pairs = []

    for row in range(len(usable) - 1):
        distances = popcount(np.bitwise_xor(dhashes[row + 1:], dhashes[row]))
        for hit in np.nonzero(distances <= MAX_DISTANCE)[0]:
            i, j = usable[row], usable[row + 1 + int(hit)]
            p_distance = bin(int(photos[i]["phash"], 16) ^ int(photos[j]["phash"], 16)).count("1")
            pairs.append((i, j, int(distances[hit]), p_distance))
        if (row + 1) % 5000 == 0:
            print(f"  compared {row + 1}/{len(usable)} photos")

    return pairs


def describe(pairs, photos):
    """Turns each pair into a readable row: which photos, which breeds, which piles."""
    rows = []
    for i, j, d_distance, p_distance in pairs:
        a, b = photos[i], photos[j]
        rows.append({
            "photo_a": a["key"],
            "photo_b": b["key"],
            "breed_a": a["breed"],
            "breed_b": b["breed"],
            "pile_a": a["pile"],
            "pile_b": b["pile"],
            "dhash_distance": d_distance,
            "phash_distance": p_distance,
            # True only when both methods agree the pair is a copy.
            "confirmed_copy": d_distance <= CONFIRM_DHASH and p_distance <= CONFIRM_PHASH,
            "exact_copy": a["sha256"] == b["sha256"],
            "same_breed": a["breed"] == b["breed"],
            "crosses_piles": a["pile"] != b["pile"],
            # e.g. "train-test", always written in the order train, val, test
            "pile_pair": "-".join(sorted((a["pile"], b["pile"]), key=PILES.index)),
        })
    rows.sort(key=lambda r: (r["dhash_distance"], r["photo_a"], r["photo_b"]))
    return rows


def sweep(rows, pile_sizes):
    """
    For each cut-off, what would be flagged as a copy. The two key columns are
    the number of test and validation photos that have a copy sitting in train:
    those are the leaked photos. pHash agreement is the share of flagged pairs
    that the second method also calls a copy at the same cut-off.
    """
    table = []
    for cutoff in SWEEP:
        flagged = [r for r in rows if r["dhash_distance"] <= cutoff]
        leaked = {"val": set(), "test": set()}
        for r in flagged:
            if r["pile_pair"] in ("train-val", "train-test"):
                other = r["photo_a"] if r["pile_a"] != "train" else r["photo_b"]
                leaked[r["pile_pair"].split("-")[1]].add(other)
        agree = sum(r["phash_distance"] <= cutoff for r in flagged)
        pile_pairs = Counter(r["pile_pair"] for r in flagged)
        table.append({
            "cutoff": cutoff,
            "pairs": len(flagged),
            "same_breed_pairs": sum(r["same_breed"] for r in flagged),
            "different_breed_pairs": sum(not r["same_breed"] for r in flagged),
            "phash_agrees_pct": round(100 * agree / len(flagged), 1) if flagged else 0.0,
            "train_val_pairs": pile_pairs["train-val"],
            "train_test_pairs": pile_pairs["train-test"],
            "val_test_pairs": pile_pairs["val-test"],
            "test_photos_with_copy_in_train": len(leaked["test"]),
            "test_leaked_pct": round(100 * len(leaked["test"]) / pile_sizes["test"], 2),
            "val_photos_with_copy_in_train": len(leaked["val"]),
            "val_leaked_pct": round(100 * len(leaked["val"]) / pile_sizes["val"], 2),
        })
    return table


# ---------------------------------------------------------------------------
# Sample sheets: pairs drawn side by side so the cut-off can be judged by eye.
# ---------------------------------------------------------------------------
THUMB = 190
GAP = 10
MARGIN = 20
CAPTION = 48
PAIRS_PER_ROW = 3


def caption_font():
    try:
        return ImageFont.load_default(size=12)
    except TypeError:  # older Pillow versions have one fixed-size default font
        return ImageFont.load_default()


def thumbnail(key):
    tile = Image.new("RGB", (THUMB, THUMB), (235, 235, 235))
    with Image.open(IMAGES_DIR / key) as img:
        small = ImageOps.contain(img.convert("RGB"), (THUMB, THUMB))
    tile.paste(small, ((THUMB - small.width) // 2, (THUMB - small.height) // 2))
    return tile


def short_breed(folder):
    return folder.split("-", 1)[-1].replace("_", " ")[:26]


def draw_sheet(sample, title, path):
    pair_width = 2 * THUMB + GAP
    cell_height = THUMB + CAPTION
    rows_needed = (len(sample) + PAIRS_PER_ROW - 1) // PAIRS_PER_ROW
    width = PAIRS_PER_ROW * pair_width + (PAIRS_PER_ROW + 1) * MARGIN
    height = 40 + rows_needed * (cell_height + MARGIN) + MARGIN

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = caption_font()
    draw.text((MARGIN, 12), title, fill="black", font=font)

    for n, r in enumerate(sample):
        x = MARGIN + (n % PAIRS_PER_ROW) * (pair_width + MARGIN)
        y = 40 + (n // PAIRS_PER_ROW) * (cell_height + MARGIN)
        sheet.paste(thumbnail(r["photo_a"]), (x, y))
        sheet.paste(thumbnail(r["photo_b"]), (x + THUMB + GAP, y))

        flags = [f"dHash {r['dhash_distance']}", f"pHash {r['phash_distance']}"]
        if r["exact_copy"]:
            flags.append("EXACT COPY")
        if not r["same_breed"]:
            flags.append("DIFFERENT BREEDS")
        colour = (170, 30, 30) if r["crosses_piles"] else (0, 0, 0)
        draw.text((x, y + THUMB + 3), f"A: {short_breed(r['breed_a'])} ({r['pile_a']})", fill=colour, font=font)
        draw.text((x, y + THUMB + 17), f"B: {short_breed(r['breed_b'])} ({r['pile_b']})", fill=colour, font=font)
        draw.text((x, y + THUMB + 31), "  ".join(flags), fill=colour, font=font)

    sheet.save(path)


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    photos = load_split()
    pile_sizes = Counter(ph["pile"] for ph in photos)

    print(f"Step 1 of 3: fingerprinting {len(photos)} photos")
    fingerprint(photos)
    unreadable = [ph["key"] for ph in photos if not ph["dhash"]]

    print("Step 2 of 3: comparing every fingerprint with every other")
    rows = describe(find_pairs(photos), photos)
    write_csv(OUTPUT_DIR / "near_duplicate_pairs.csv", PAIR_FIELDS, rows)
    table = sweep(rows, pile_sizes)
    write_csv(OUTPUT_DIR / "threshold_sweep.csv", list(table[0].keys()), table)

    # Apply the chosen two-method rule.
    confirmed = [r for r in rows if r["confirmed_copy"]]
    write_csv(OUTPUT_DIR / "confirmed_copies.csv", PAIR_FIELDS, confirmed)

    # The exclusion list: validation and test photos that have a confirmed copy
    # in training. Evaluating with and without these gives the leaked score and
    # the clean score.
    leaked_rows = []
    for r in confirmed:
        if r["pile_pair"] not in ("train-val", "train-test"):
            continue
        held, trained = ("a", "b") if r["pile_a"] != "train" else ("b", "a")
        leaked_rows.append({
            "photo": r[f"photo_{held}"],
            "pile": r[f"pile_{held}"],
            "breed": r[f"breed_{held}"],
            "copy_in_train": r[f"photo_{trained}"],
            "copy_breed": r[f"breed_{trained}"],
            "dhash_distance": r["dhash_distance"],
            "phash_distance": r["phash_distance"],
            "exact_copy": r["exact_copy"],
            "same_breed": r["same_breed"],
        })
    leaked_rows.sort(key=lambda r: (r["pile"], r["photo"]))
    write_csv(OUTPUT_DIR / "leaked_photos.csv", LEAK_FIELDS, leaked_rows)

    # Confirmed copies carrying two different breed labels. One label in each
    # pair must be wrong, which is evidence on label quality for the report.
    conflicts = [r for r in confirmed if not r["same_breed"]]
    write_csv(OUTPUT_DIR / "label_conflicts.csv", PAIR_FIELDS, conflicts)

    print("Step 3 of 3: drawing sample sheets")
    rng = random.Random(SEED)
    for low, high in SHEET_BANDS:
        band = [r for r in rows if low <= r["dhash_distance"] <= high]
        if not band:
            continue
        sample = sorted(rng.sample(band, min(PAIRS_PER_SHEET, len(band))),
                        key=lambda r: r["dhash_distance"])
        name = f"{low}" if low == high else f"{low}-{high}"
        draw_sheet(
            sample,
            f"dHash distance {name}: {len(sample)} of {len(band)} pairs shown. "
            f"Red caption = the two photos are in different piles.",
            OUTPUT_DIR / f"pairs_distance_{name}.png",
        )

    # One sheet of the pairs that matter most: confirmed copies split across piles.
    crossing = [r for r in confirmed if r["crosses_piles"]]
    if crossing:
        sample = sorted(rng.sample(crossing, min(PAIRS_PER_SHEET, len(crossing))),
                        key=lambda r: r["dhash_distance"])
        draw_sheet(
            sample,
            f"Confirmed copies sitting in different piles: {len(sample)} of {len(crossing)} shown. "
            f"Rule: dHash {CONFIRM_DHASH} or less and pHash {CONFIRM_PHASH} or less.",
            OUTPUT_DIR / "pairs_confirmed_crossing.png",
        )

    # ----- summary -----
    copies = Counter(ph["sha256"] for ph in photos)
    exact_groups = sum(1 for c in copies.values() if c > 1)
    exact_extra = sum(c - 1 for c in copies.values() if c > 1)
    exact_crossing = sum(1 for r in rows if r["exact_copy"] and r["crosses_piles"])

    lines = [
        "NEAR-DUPLICATE CHECK ON THE SHARED SPLIT",
        "=" * 100,
        f"Photos fingerprinted     {len(photos) - len(unreadable)} of {len(photos)}"
        f" ({len(unreadable)} could not be read)",
        f"Exact file copies        {exact_groups} groups of identical files, "
        f"{exact_extra} extra copies, {exact_crossing} exact pairs in different piles",
        f"Pairs within distance {MAX_DISTANCE}  {len(rows)}",
        "",
        "WHAT EACH CUT-OFF WOULD FLAG AS A COPY",
        f"{'cut-off':>7} {'pairs':>6} {'same':>6} {'diff':>6} {'pHash':>7} "
        f"{'tr-val':>7} {'tr-test':>8} {'val-test':>9} {'test leaked':>16} {'val leaked':>16}",
        f"{'':>7} {'':>6} {'breed':>6} {'breed':>6} {'agrees':>7} "
        f"{'pairs':>7} {'pairs':>8} {'pairs':>9} {'photos (%)':>16} {'photos (%)':>16}",
    ]
    for t in table:
        lines.append(
            f"{t['cutoff']:>7} {t['pairs']:>6} {t['same_breed_pairs']:>6} "
            f"{t['different_breed_pairs']:>6} {t['phash_agrees_pct']:>6}% "
            f"{t['train_val_pairs']:>7} {t['train_test_pairs']:>8} {t['val_test_pairs']:>9} "
            f"{t['test_photos_with_copy_in_train']:>8} ({t['test_leaked_pct']:>5}%) "
            f"{t['val_photos_with_copy_in_train']:>8} ({t['val_leaked_pct']:>5}%)"
        )
    leaked_by_pile = Counter(r["pile"] for r in leaked_rows)
    conflict_breeds = {b for r in conflicts for b in (r["breed_a"], r["breed_b"])}
    lines += [
        "",
        f"THE CHOSEN RULE: both methods agreeing, dHash {CONFIRM_DHASH} or less "
        f"and pHash {CONFIRM_PHASH} or less",
        f"  Confirmed copies             {len(confirmed)} pairs, "
        f"{sum(r['crosses_piles'] for r in confirmed)} of them in different piles",
        f"  Test photos with a copy in training        {leaked_by_pile['test']} "
        f"({round(100 * leaked_by_pile['test'] / pile_sizes['test'], 2)} per cent of the test pile)",
        f"  Validation photos with a copy in training  {leaked_by_pile['val']} "
        f"({round(100 * leaked_by_pile['val'] / pile_sizes['val'], 2)} per cent of the validation pile)",
        f"  Same photograph, two breed labels          {len(conflicts)} pairs, "
        f"touching {len(conflict_breeds)} breeds",
        "  Saved as confirmed_copies.csv, leaked_photos.csv and label_conflicts.csv.",
        "  leaked_photos.csv is the exclusion list for reporting a clean test score.",
        "",
        "Reading the table:",
        "  test leaked = test photos that have a look-alike in training. These are the leak.",
        "  diff breed  = the same picture filed under two different breeds, so one label is wrong.",
        "  pHash agrees = share of flagged pairs the second method also calls a copy.",
        "                 Where this falls away sharply, the cut-off has become too loose.",
        "",
        "Next: look at the pairs_distance_*.png sheets and choose the cut-off.",
    ]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "duplicate_check_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print()
    print(summary)
    print(f"\nWrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
