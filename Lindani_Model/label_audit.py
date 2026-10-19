"""
label_audit.py

Stage 3: a sample audit of the breed labels.

The labels in this dataset are inherited from ImageNet and, as far as either the
Stanford page or the Kaggle page records, were never verified. This script draws
a fixed random sample of 100 photographs so their labels can be checked by eye,
and then reports the observed disagreement rate.

Run it twice.

    python Lindani_Model/label_audit.py

The first run picks the sample with seed 42, draws five numbered sheets, and
creates label_audit_tally.csv with an empty verdict column.

Then look at the sheets and fill in the verdict column in that file, one of:

    agree      the photograph does look like the breed on the label
    disagree   it does not, and the label looks wrong
    unsure     cannot tell, or the breed is unfamiliar

Running it again reads the verdicts and writes the summary, including a 95 per
cent confidence interval, because 100 photographs is a small sample and the rate
should be reported with its uncertainty.

An existing tally file is never overwritten, so filled-in verdicts are safe.

Honest note for the report: the person judging is not a breed expert. That is why
"unsure" exists and why the unsure count is reported rather than hidden. The
examples figure in results/dataset_figures/fig_class_examples.png is a useful
reference while judging, and looking breeds up while judging is expected.

Needs: matplotlib and pillow, already installed for the figures.
"""

import csv
import math
import random

import matplotlib.pyplot as plt

from dataset_figures import INK, load_photos, square_tile
from split_check import LINDANI_DIR

OUTPUT_DIR = LINDANI_DIR / "results" / "label_audit"
TALLY_PATH = OUTPUT_DIR / "label_audit_tally.csv"

SAMPLE_SIZE = 100
SHEET_COLS, SHEET_ROWS = 4, 5      # 20 photographs per sheet, so five sheets
SEED = 42
VERDICTS = ("agree", "disagree", "unsure")
TALLY_FIELDS = ["number", "verdict", "note", "breed_name", "pile", "breed_folder", "photo"]


def wilson_interval(successes, trials, z=1.96):
    """
    A 95 per cent confidence interval for a proportion, using the Wilson method.
    It behaves sensibly on small samples and near 0 per cent, where the textbook
    interval can run below zero.
    """
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def draw_sheets(sample):
    """
    Numbered contact sheets, 20 photographs each. Each tile shows its number and
    the breed the dataset claims, which is what is being judged. The whole
    photograph is shown, padded rather than cropped, so nothing that might
    matter is cut off.
    """
    per_sheet = SHEET_COLS * SHEET_ROWS
    for start in range(0, len(sample), per_sheet):
        block = sample[start:start + per_sheet]
        sheet_number = start // per_sheet + 1

        fig, axes = plt.subplots(SHEET_ROWS, SHEET_COLS, figsize=(13, 17))
        for ax, row in zip(axes.ravel(), block):
            ax.imshow(square_tile(row["photo"]))
            ax.set_title(f"{row['number']}. {row['breed_name']}", fontsize=10, color=INK, pad=4)
        for ax in axes.ravel():
            ax.axis("off")

        fig.suptitle(
            f"Label audit sheet {sheet_number}: photographs "
            f"{block[0]['number']} to {block[-1]['number']}. "
            "Does each photograph match the breed named above it?",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(OUTPUT_DIR / f"audit_sheet_{sheet_number}.png")
        plt.close(fig)


def build_sample():
    """The fixed sample. Seed 42, so the same 100 photographs every time."""
    photos, names = load_photos()
    chosen = random.Random(SEED).sample(photos, SAMPLE_SIZE)
    sample = [
        {
            "number": n,
            "verdict": "",
            "note": "",
            "breed_name": names.get(ph["breed"], ph["breed"]),
            "pile": ph["pile"],
            "breed_folder": ph["breed"],
            "photo": ph["key"],
        }
        for n, ph in enumerate(chosen, 1)
    ]
    return sample


def score(rows):
    """Reads the filled-in verdicts and writes the summary for the report."""
    counts = {verdict: 0 for verdict in VERDICTS}
    unrecognised, blank = [], 0
    for row in rows:
        verdict = row["verdict"].strip().lower()
        if not verdict:
            blank += 1
        elif verdict in counts:
            counts[verdict] += 1
        else:
            unrecognised.append((row["number"], row["verdict"]))

    judged = counts["agree"] + counts["disagree"]
    lines = [
        "LABEL AUDIT, SAMPLE OF 100 PHOTOGRAPHS",
        "=" * 72,
        f"Sample                   {len(rows)} photographs, drawn with seed {SEED}",
        f"Verdicts recorded        {len(rows) - blank}",
        f"Still blank              {blank}",
        "",
        f"  Agree                  {counts['agree']}",
        f"  Disagree               {counts['disagree']}",
        f"  Unsure                 {counts['unsure']}",
    ]

    if unrecognised:
        lines.append("")
        lines.append("Entries not understood, expected agree, disagree or unsure:")
        lines += [f"  photograph {number}: {value!r}" for number, value in unrecognised]

    if judged:
        rate = counts["disagree"] / judged
        low, high = wilson_interval(counts["disagree"], judged)
        lower_bound = counts["disagree"] / len(rows)
        lines += [
            "",
            "OBSERVED LABEL DISAGREEMENT",
            f"  Excluding unsure       {counts['disagree']} of {judged} "
            f"({100 * rate:.1f} per cent)",
            f"  95 per cent interval   {100 * low:.1f} to {100 * high:.1f} per cent",
            f"  Of the whole sample    {counts['disagree']} of {len(rows)} "
            f"({100 * lower_bound:.1f} per cent), treating every unsure as correct",
            "",
            "Both figures belong in the report. The first is the rate among the",
            "photographs that could be judged. The second is the cautious floor.",
            "The interval is wide because the sample is small, and saying so is",
            "better than quoting a single number as though it were precise.",
        ]
        if counts["disagree"]:
            lines += ["", "Disagreements, for the report or a figure:"]
            lines += [
                f"  {row['number']:>3}. {row['breed_name']} ({row['pile']}) "
                f"{row['photo']}" + (f"  note: {row['note']}" if row["note"].strip() else "")
                for row in rows
                if row["verdict"].strip().lower() == "disagree"
            ]
    else:
        lines += ["", "No verdicts recorded yet, so there is nothing to score."]

    summary = "\n".join(lines)
    (OUTPUT_DIR / "label_audit_summary.txt").write_text(summary + "\n", encoding="utf-8")
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = build_sample()

    if TALLY_PATH.exists():
        with open(TALLY_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Guard against scoring a tally that belongs to a different sample.
        if {row["photo"] for row in rows} != {row["photo"] for row in sample}:
            print(f"{TALLY_PATH.name} does not match the sample this script draws.")
            print("It has been left untouched. Move it aside if you want a fresh sample.")
            return

        print(f"Reading verdicts from {TALLY_PATH.name}")
        print()
        print(score(rows))
        print(f"\nWrote the summary to {OUTPUT_DIR}")
        return

    with open(TALLY_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TALLY_FIELDS)
        writer.writeheader()
        writer.writerows(sample)

    print(f"Drawing {len(sample)} photographs onto sheets")
    draw_sheets(sample)

    print(f"\nWrote {OUTPUT_DIR}")
    print("\nWhat to do now:")
    print("  1. Open audit_sheet_1.png to audit_sheet_5.png.")
    print("  2. For each numbered photograph, decide whether it matches the breed named above it.")
    print(f"  3. Fill in the verdict column of {TALLY_PATH.name} with agree, disagree or unsure.")
    print("     The note column is optional, for what you thought it actually was.")
    print("  4. Run this script again to get the rate and the confidence interval.")
    print("\nfig_class_examples.png in results/dataset_figures is a useful reference while judging.")


if __name__ == "__main__":
    main()
