"""
dataset_figures.py

Stage 3: the dataset figures and the per-breed count table for the report.

Reads the photographs and the shared split without changing either, and produces:

  fig_class_examples.png      one training photograph from each of the 120 breeds
  fig_smallest_subjects.png   photographs where the dog fills least of the frame
  fig_class_distribution.png  photographs per breed, and how breed sizes cluster
  fig_image_sizes.png         photograph dimensions and shapes against the 224 px input
  table_class_counts.csv      counts per breed, per pile, before and after cleaning
  dataset_summary.txt         the figures written out as text, ready to quote
  image_sizes.csv             the measured width, height and colour mode per photo
  bounding_boxes.csv          how many dogs are boxed, and how much frame they fill

Run from the repository root (the Dog_Classifier folder), with the virtual
environment active:
    python Lindani_Model/dataset_figures.py

The first run reads the header of all 20,580 files to get their dimensions,
about a minute. Those are saved, so later runs are quick. Delete image_sizes.csv
to force a fresh pass.

Needs: pip install matplotlib
"""

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # draw straight to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from split_check import IMAGES_DIR, LINDANI_DIR, MANIFEST_PATH, PILES, REPO_ROOT, local_key

OUTPUT_DIR = LINDANI_DIR / "results" / "dataset_figures"
SIZES_PATH = OUTPUT_DIR / "image_sizes.csv"
BOXES_PATH = OUTPUT_DIR / "bounding_boxes.csv"
DUPLICATE_DIR = LINDANI_DIR / "results" / "duplicate_check"
ANNOTATIONS_DIR = REPO_ROOT / "archive" / "annotations" / "Annotation"

INPUT_SIZE = 224                 # the input size all three models use
GRID_COLS, GRID_ROWS = 10, 12    # 120 breeds
THUMB = 200                      # thumbnail size in the examples figure
SCATTER_CLIP = 1200              # dimensions above this are left off the scatter
SHAPE_RANGE = (0.4, 2.5)         # shapes outside this are counted, not plotted
SMALLEST_TILES, SMALLEST_COLS = 20, 5  # the limitation figure
# The examples figure aims for a photograph where the dog dominates the frame but
# is still shown whole. The largest box of all tends to be an extreme head crop,
# which hides the build and proportions that separate similar breeds.
TARGET_BOX_FRACTION = 0.65

TABLE_FIELDS = [
    "label", "breed_folder", "breed_name", "total_images", "train", "val", "test",
    "confirmed_duplicate_pairs", "test_photos_excluded", "test_after_exclusion",
]

# One colour for the data and grey for anything structural. A second colour is
# used only where a text label also names the thing, so nothing depends on
# colour alone. Both read correctly in print and in greyscale.
DATA = "#2F6F94"
ACCENT = "#B4532A"
MUTED = "#6B7780"
INK = "#1A1A1A"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": "#D7DCE0",
    "grid.linewidth": 0.6,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def load_photos():
    """
    One entry per photo: short name, breed folder and pile, sorted by name, plus
    the readable breed names. This mirrors load_split in duplicate_check.py, kept
    separate so the figures do not depend on the fingerprinting libraries.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    photos = [
        {"key": local_key(entry), "breed": entry["breed_folder"], "pile": pile}
        for pile in PILES
        for entry in manifest["splits"][pile]
    ]
    names = {c["folder"]: c["display_name"] for c in manifest["classes"].values()}
    return sorted(photos, key=lambda ph: ph["key"]), names


def measure(photos):
    """
    Adds width, height and colour mode to every photo. Only each file's header is
    read, not the whole picture, which is why this takes about a minute rather
    than ten. Results are cached in image_sizes.csv.
    """
    if SIZES_PATH.exists():
        with open(SIZES_PATH, newline="", encoding="utf-8") as f:
            saved = {row["key"]: row for row in csv.DictReader(f)}
        if set(saved) == {ph["key"] for ph in photos}:
            for ph in photos:
                row = saved[ph["key"]]
                ph.update(width=int(row["width"]), height=int(row["height"]), mode=row["mode"])
            print(f"Loaded saved image sizes for {len(photos)} photos.")
            return
        print("Saved image sizes do not match the split, so measuring again.")

    for n, ph in enumerate(photos, 1):
        with Image.open(IMAGES_DIR / ph["key"]) as img:
            ph["width"], ph["height"] = img.size
            ph["mode"] = img.mode
        if n % 5000 == 0 or n == len(photos):
            print(f"  measured {n}/{len(photos)}")

    with open(SIZES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["key", "breed", "pile", "width", "height", "mode"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(photos)


def read_boxes(photos):
    """
    Reads the box drawn around each dog from the annotation files supplied with
    the dataset: one small XML file per photograph, giving the frame size and one
    box per dog.

    Adds two things to every photo: how many dogs are boxed, and what share of
    the frame the largest box covers. The share decides which photograph
    represents each breed in the examples figure, so that every tile actually
    shows a dog. Cached in bounding_boxes.csv.
    """
    if BOXES_PATH.exists():
        with open(BOXES_PATH, newline="", encoding="utf-8") as f:
            saved = {row["key"]: row for row in csv.DictReader(f)}
        if set(saved) == {ph["key"] for ph in photos}:
            for ph in photos:
                row = saved[ph["key"]]
                ph.update(dogs=int(row["dogs"]), box_fraction=float(row["box_fraction"]))
            print(f"Loaded saved bounding boxes for {len(photos)} photos.")
            return
        print("Saved bounding boxes do not match the split, so reading them again.")

    for n, ph in enumerate(photos, 1):
        # The annotation file has the same name as the photograph, without .jpg
        root = ET.parse(ANNOTATIONS_DIR / ph["key"].rsplit(".", 1)[0]).getroot()
        size = root.find("size")
        frame = int(size.findtext("width")) * int(size.findtext("height"))
        fractions = []
        for box in root.iter("bndbox"):
            wide = int(box.findtext("xmax")) - int(box.findtext("xmin"))
            tall = int(box.findtext("ymax")) - int(box.findtext("ymin"))
            fractions.append(wide * tall / frame if frame else 0.0)
        ph["dogs"] = len(fractions)
        ph["box_fraction"] = round(max(fractions), 4) if fractions else 0.0
        if n % 5000 == 0 or n == len(photos):
            print(f"  read {n}/{len(photos)} annotations")

    with open(BOXES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "dogs", "box_fraction"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(photos)


def duplicate_counts():
    """
    Reads what duplicate_check.py found: how many confirmed duplicate pairs touch
    each breed, and how many test photos each breed loses to the leak list.
    Returns empty counts if those files have not been produced yet.
    """
    pairs, excluded = Counter(), Counter()

    confirmed_path = DUPLICATE_DIR / "confirmed_copies.csv"
    if confirmed_path.exists():
        with open(confirmed_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pairs[row["breed_a"]] += 1
                if row["breed_b"] != row["breed_a"]:
                    pairs[row["breed_b"]] += 1

    leaked_path = DUPLICATE_DIR / "leaked_photos.csv"
    if leaked_path.exists():
        with open(leaked_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["pile"] == "test":
                    excluded[row["breed"]] += 1

    return pairs, excluded


def build_table(photos, names, pairs, excluded):
    """
    One row per breed: how many photographs it has, how they are split, how many
    confirmed duplicate pairs touch it, and how many test photographs are set
    aside as leaked. The last column is the effective test count.
    """
    per_breed = Counter(ph["breed"] for ph in photos)
    per_pile = {p: Counter(ph["breed"] for ph in photos if ph["pile"] == p) for p in PILES}

    rows = []
    for label, folder in enumerate(sorted(per_breed)):
        test = per_pile["test"][folder]
        rows.append({
            "label": label,
            "breed_folder": folder,
            "breed_name": names.get(folder, folder),
            "total_images": per_breed[folder],
            "train": per_pile["train"][folder],
            "val": per_pile["val"][folder],
            "test": test,
            "confirmed_duplicate_pairs": pairs[folder],
            "test_photos_excluded": excluded[folder],
            "test_after_exclusion": test - excluded[folder],
        })

    total = {
        "label": "", "breed_folder": "TOTAL", "breed_name": "All breeds",
        **{field: sum(r[field] for r in rows) for field in TABLE_FIELDS[3:]},
    }
    with open(OUTPUT_DIR / "table_class_counts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows + [total])

    return rows, total


def square_tile(key, pad=(247, 247, 247)):
    """
    Fits a whole photograph into a square tile, padding the short side rather
    than cropping it, so the tiles line up without cutting anything out of the
    picture.
    """
    tile = Image.new("RGB", (THUMB, THUMB), pad)
    with Image.open(IMAGES_DIR / key) as img:
        small = ImageOps.contain(img.convert("RGB"), (THUMB, THUMB))
    tile.paste(small, ((THUMB - small.width) // 2, (THUMB - small.height) // 2))
    return tile


def figure_examples(photos, names):
    """
    One photograph per breed in a 10 by 12 grid. For each breed the photograph
    chosen is the training one whose boxed dog covers closest to
    TARGET_BOX_FRACTION of the frame, so the dog dominates the tile but is still
    shown whole. Ties break alphabetically, so the figure is identical every run,
    and nothing from validation or test appears.
    """
    best = {}
    for ph in photos:  # already sorted by name, so ties break alphabetically
        if ph["pile"] != "train":
            continue
        distance = abs(ph["box_fraction"] - TARGET_BOX_FRACTION)
        current = best.get(ph["breed"])
        if current is None or distance < current[0]:
            best[ph["breed"]] = (distance, ph["key"])
    folders = sorted(best)

    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(16, 21))
    for ax, folder in zip(axes.ravel(), folders):
        ax.imshow(square_tile(best[folder][1]))
        ax.set_title(names.get(folder, folder), fontsize=6.5, color=INK, pad=2)
    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"One training photograph from each of the {len(folders)} breeds, choosing one where "
        f"the dog covers about {100 * TARGET_BOX_FRACTION:.0f} per cent of the frame",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUTPUT_DIR / "fig_class_examples.png")
    plt.close(fig)


def figure_smallest_subjects(photos, names):
    """
    The opposite end of the same measurement: the training photographs where the
    boxed dog fills the least of the frame. This is a limitation figure. It shows
    that some photographs are dominated by people, vehicles or scenery, so part
    of the error any model makes comes from the data rather than the model.
    """
    # One photograph per breed, so the figure shows the problem across 20 breeds
    # rather than four photographs of the same dog show.
    smallest, seen = [], set()
    for ph in sorted((p for p in photos if p["pile"] == "train"),
                     key=lambda p: (p["box_fraction"], p["key"])):
        if ph["breed"] in seen:
            continue
        seen.add(ph["breed"])
        smallest.append(ph)
        if len(smallest) == SMALLEST_TILES:
            break
    rows_needed = -(-len(smallest) // SMALLEST_COLS)

    fig, axes = plt.subplots(rows_needed, SMALLEST_COLS, figsize=(14, 3.4 * rows_needed))
    for ax, ph in zip(axes.ravel(), smallest):
        ax.imshow(square_tile(ph["key"]))
        several = "" if ph["dogs"] == 1 else f", {ph['dogs']} dogs boxed"
        ax.set_title(
            f"{names.get(ph['breed'], ph['breed'])}\n"
            f"dog covers {100 * ph['box_fraction']:.1f} per cent{several}",
            fontsize=7.5, color=INK, pad=3,
        )
    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("Photographs where the boxed dog fills the least of the frame", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_DIR / "fig_smallest_subjects.png")
    plt.close(fig)


def figure_distribution(rows):
    """Left: photographs per breed, sorted. Right: how breed sizes cluster."""
    counts = np.array([r["total_images"] for r in rows])
    names = [r["breed_name"] for r in rows]
    order = np.argsort(-counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2, 1]})

    ax1.bar(range(len(counts)), counts[order], color=DATA, width=0.85, linewidth=0)
    ax1.set_xlim(-1, len(counts))
    ax1.set_xticks([])
    ax1.set_xlabel(f"the {len(counts)} breeds, ordered from most to fewest photographs")
    ax1.set_ylabel("photographs")
    ax1.set_title("Photographs per breed")
    ax1.yaxis.grid(True)
    ax1.set_axisbelow(True)

    top, bottom = order[0], order[-1]
    ax1.annotate(
        f"{names[top]}, {counts[top]}",
        xy=(0, counts[top]), xytext=(8, counts[top] + 14), fontsize=8, color=ACCENT,
        arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.8),
    )
    ax1.annotate(
        f"{names[bottom]}, {counts[bottom]}",
        xy=(len(counts) - 1, counts[bottom]), xytext=(len(counts) - 46, counts[bottom] + 34),
        fontsize=8, color=ACCENT,
        arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=0.8),
    )
    ax1.text(
        0.99, 0.94, f"imbalance ratio {counts.max() / counts.min():.2f} to 1",
        transform=ax1.transAxes, ha="right", va="top", fontsize=8, color=MUTED,
    )

    ax2.hist(counts, bins=14, color=DATA, linewidth=0)
    # Mean and median together, because the shape is skewed: most breeds sit near
    # the minimum, so the mean alone would misdescribe the dataset.
    ax2.axvline(counts.mean(), color=ACCENT, linestyle="--", linewidth=1)
    ax2.axvline(np.median(counts), color=MUTED, linestyle=":", linewidth=1.2)
    # Both labels sit in the empty right-hand space, so neither covers a bar.
    ax2.text(0.97, 0.94, f"- -  mean {counts.mean():.0f}", transform=ax2.transAxes,
             ha="right", va="top", fontsize=8, color=ACCENT)
    ax2.text(0.97, 0.85, f"···  median {np.median(counts):.0f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
    ax2.set_xlabel("photographs per breed")
    ax2.set_ylabel("breeds")
    ax2.set_title("How breed sizes cluster")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_class_distribution.png")
    plt.close(fig)


def figure_sizes(photos):
    """Left: dimensions against the 224 px input size. Right: the spread of shapes."""
    widths = np.array([ph["width"] for ph in photos])
    heights = np.array([ph["height"] for ph in photos])
    ratios = widths / heights
    shown = (widths <= SCATTER_CLIP) & (heights <= SCATTER_CLIP)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    ax1.scatter(widths[shown], heights[shown], s=3, alpha=0.12, color=DATA, linewidths=0)
    ax1.axvline(INPUT_SIZE, color=ACCENT, linestyle="--", linewidth=1)
    ax1.axhline(INPUT_SIZE, color=ACCENT, linestyle="--", linewidth=1)
    ax1.text(INPUT_SIZE + 14, SCATTER_CLIP * 0.93, f"{INPUT_SIZE} px, the model input size",
             fontsize=8, color=ACCENT)
    ax1.set_xlim(0, SCATTER_CLIP)
    ax1.set_ylim(0, SCATTER_CLIP)
    ax1.set_xlabel("width, pixels")
    ax1.set_ylabel("height, pixels")
    ax1.set_title(f"Dimensions ({int((~shown).sum())} larger than {SCATTER_CLIP} px not shown)")
    ax1.grid(True)
    ax1.set_axisbelow(True)

    # Extreme shapes are counted and named rather than piled into the end bins,
    # which would invent a spike that is not in the data.
    inside = (ratios >= SHAPE_RANGE[0]) & (ratios <= SHAPE_RANGE[1])
    ax2.hist(ratios[inside], bins=50, color=DATA, linewidth=0)
    ax2.axvline(1.0, color=ACCENT, linestyle="--", linewidth=1)
    ax2.text(1.04, ax2.get_ylim()[1] * 0.9, "square", fontsize=8, color=ACCENT)
    ax2.set_xlim(*SHAPE_RANGE)
    ax2.set_xlabel("shape, width divided by height")
    ax2.set_ylabel("photographs")
    ax2.set_title(f"Shapes ({int((~inside).sum())} more extreme than the range not shown)")
    ax2.yaxis.grid(True)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_image_sizes.png")
    plt.close(fig)


def write_summary(photos, rows, total):
    """The numbers behind the figures, in text, ready to quote in the report."""
    counts = np.array([r["total_images"] for r in rows])
    widths = np.array([ph["width"] for ph in photos])
    heights = np.array([ph["height"] for ph in photos])
    ratios = widths / heights
    smaller = int((np.minimum(widths, heights) < INPUT_SIZE).sum())
    modes = Counter(ph["mode"] for ph in photos)
    fractions = np.array([ph["box_fraction"] for ph in photos])
    dogs = np.array([ph["dogs"] for ph in photos])

    def share(n):
        return f"{n} ({100 * n / len(photos):.2f} per cent)"

    fewest = rows[int(np.argmin(counts))]["breed_name"]
    most = rows[int(np.argmax(counts))]["breed_name"]

    lines = [
        "STANFORD DOGS, DATASET DESCRIPTION FOR THE REPORT",
        "=" * 72,
        f"Photographs                {len(photos)}",
        f"Breeds                     {len(rows)}",
        f"Colour modes               {dict(modes)}",
        "",
        "PHOTOGRAPHS PER BREED",
        f"  Fewest                   {counts.min()} ({fewest})",
        f"  Most                     {counts.max()} ({most})",
        f"  Mean                     {counts.mean():.1f}",
        f"  Median                   {np.median(counts):.1f}",
        f"  Imbalance ratio          {counts.max() / counts.min():.2f} to 1",
        "",
        "THE SPLIT, AS SHARED BY THE GROUP",
        f"  Train                    {total['train']}",
        f"  Validation               {total['val']}",
        f"  Test                     {total['test']}",
        "",
        "AFTER CLEANING",
        "  No photograph is deleted. The confirmed duplicates found by",
        "  duplicate_check.py are recorded, and the leaked test photographs are",
        "  set aside so a clean test score can be reported alongside the full one.",
        f"  Confirmed duplicate pair mentions per breed, total   {total['confirmed_duplicate_pairs']}",
        f"  Test photographs set aside as leaked                 {total['test_photos_excluded']}",
        f"  Effective test set                                   {total['test_after_exclusion']}",
        "",
        "DIMENSIONS",
        f"  Width                    {widths.min()} to {widths.max()} px, median {int(np.median(widths))}",
        f"  Height                   {heights.min()} to {heights.max()} px, median {int(np.median(heights))}",
        f"  Smaller than {INPUT_SIZE} px on at least one side   {share(smaller)}",
        "    These are enlarged to reach the model input size, which softens them.",
        "",
        "SHAPES",
        f"  Landscape                {share(int((ratios > 1.05).sum()))}",
        f"  Portrait                 {share(int((ratios < 0.95).sum()))}",
        f"  Near square              {share(int(((ratios >= 0.95) & (ratios <= 1.05)).sum()))}",
        f"  Median shape             {np.median(ratios):.2f} wide for every 1 tall",
        "    The shared pipeline resizes straight to 224 by 224, so photographs that",
        "    are not square are squashed rather than cropped. Worth stating in the",
        "    methodology, since it distorts the shape of the dog.",
        "",
        "WHERE THE DOG SITS IN THE FRAME, FROM THE SUPPLIED BOXES",
        f"  Median share of the frame covered by the dog   {100 * np.median(fractions):.1f} per cent",
        f"  Dog covers less than a quarter of the frame    {share(int((fractions < 0.25).sum()))}",
        f"  Smallest single subject                        "
        f"{100 * fractions.min():.1f} per cent of the frame",
        f"  More than one dog under one breed label       {share(int((dogs > 1).sum()))}, "
        f"up to {dogs.max()} in one photograph",
        "    Photographs where the dog is small, and photographs where several dogs",
        "    share a single label, are limitations of the data rather than of any",
        "    model. See fig_smallest_subjects.png.",
        "",
        "FIGURES",
        "  fig_class_examples.png      one training photograph from every breed",
        "  fig_smallest_subjects.png   the photographs where the dog fills least of the frame",
        "  fig_class_distribution.png  photographs per breed, and how breed sizes cluster",
        "  fig_image_sizes.png         dimensions and shapes against the 224 px input",
        "  table_class_counts.csv      the per-breed table, before and after cleaning",
    ]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "dataset_summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    photos, names = load_photos()

    print(f"Step 1 of 4: measuring {len(photos)} photographs")
    measure(photos)

    print("Step 2 of 4: reading the supplied bounding boxes")
    read_boxes(photos)

    print("Step 3 of 4: building the per-breed table")
    pairs, excluded = duplicate_counts()
    rows, total = build_table(photos, names, pairs, excluded)

    print("Step 4 of 4: drawing the figures")
    figure_examples(photos, names)
    figure_smallest_subjects(photos, names)
    figure_distribution(rows)
    figure_sizes(photos)

    print()
    print(write_summary(photos, rows, total))
    print(f"\nWrote results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
