"""
make_architecture_figure.py - draws swin_architecture.png for the report.

Swin-T as trained with JP's two-phase transfer learning recipe: which parts
are frozen, which train in phase 1 and phase 2, plus how shifted windows and
a pair of Swin blocks work. Parameter counts are measured from torchvision's
swin_t with the new 120-class head (114-class head: 4,614 fewer weights).

Run from the repository root:
    python Sulaiman_Model/make_architecture_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "swin_architecture.png"

INK = "#1F3A5F"
MUTED = "#6B7780"
FROZEN_FILL, FROZEN_EDGE = "#E9EDF1", "#8A97A6"
TRAIN_FILL, TRAIN_EDGE = "#DCEBF7", "#2F6F94"
HEAD_FILL, HEAD_EDGE = "#FBE9E1", "#B4532A"
WINDOW_COLOURS = ["#F6E3D8", "#ECE4F2", "#DCE9F4", "#E3EFDA", "#FFF2CC", "#EAF4F4", "#F3F1DC", "#F9E0E0", "#E8E8F5"]

# measured with torchvision 0.26 (see notebook section 2)
P_EMBED, P_S1, P_M1, P_S2, P_M2, P_S3, P_M3, P_S4 = 4896, 224694, 74496, 891756, 296448, 10658952, 1182720, 14183856
P_NORM, P_HEAD = 1536, 92280
TOTAL = sum([P_EMBED, P_S1, P_M1, P_S2, P_M2, P_S3, P_M3, P_S4, P_NORM, P_HEAD])
PHASE2 = P_M3 + P_S4 + P_NORM + P_HEAD
FROZEN_ALWAYS = TOTAL - PHASE2

fig = plt.figure(figsize=(16, 9.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160); ax.set_ylim(0, 96); ax.axis("off")


def box(x, y, w, h, title, lines, fill, edge, title_colour=INK, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=title_colour)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 7.6 - 2.6 * i, line, ha="center", va="center", fontsize=9.5, color=MUTED)


def arrow(x0, y0, x1, y1, colour=MUTED, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=colour, lw=lw))


# ---------------------------------------------------------------- title
ax.text(80, 92.5, "Swin-T (Swin Transformer Tiny) as trained in this project", ha="center",
        fontsize=19, fontweight="bold", color=INK)
ax.text(80, 88.6, "Pretrained on ImageNet-1K, new head for 120 breeds (114 on the simplified split), "
        "two-phase transfer learning (JP's recipe)", ha="center", fontsize=11.5, color=MUTED)

# ---------------------------------------------------------------- top row
y, h = 62, 19
ax.text(6.5, y + h - 1.5, "Photo", ha="center", fontsize=12, fontweight="bold", color=INK)
for i in range(5):
    ax.plot([2.5 + 2 * i, 2.5 + 2 * i], [y + 3, y + 11], color="#C9D1DA", lw=0.8)
    ax.plot([2.5, 10.5], [y + 3 + 2 * i, y + 3 + 2 * i], color="#C9D1DA", lw=0.8)
ax.add_patch(Rectangle((2.5, y + 3), 8, 8, fill=False, ec=MUTED, lw=1.2))
ax.text(6.5, y + 0.5, "224 × 224 × 3", ha="center", fontsize=9, color=MUTED)

stages = [
    (14, "Patch partition", ["4 × 4 pixel patches", "linear embedding", "56 × 56 grid, 96 dims"], False),
    (36, "Stage 1", ["2 Swin blocks", "3 heads", "56 × 56 × 96"], False),
    (58, "Stage 2", ["2 Swin blocks", "6 heads", "28 × 28 × 192"], False),
    (80, "Stage 3", ["6 Swin blocks", "12 heads", "14 × 14 × 384"], False),
    (102, "Stage 4", ["2 Swin blocks", "24 heads", "7 × 7 × 768"], True),
]
w = 18
for x, title, lines, trainable in stages:
    box(x, y, w, h, title, lines, TRAIN_FILL if trainable else FROZEN_FILL,
        TRAIN_EDGE if trainable else FROZEN_EDGE)
box(126, y, 30, h, "New head", ["input: final LayerNorm + average pool → 768", "Dropout 0.3",
                               "Linear 768 → 120 (or 114)"], HEAD_FILL, HEAD_EDGE, title_colour=HEAD_EDGE)

arrow(11, y + h / 2, 13.6, y + h / 2)
for x0 in (32, 54, 76, 98, 120):
    arrow(x0 + 0.2, y + h / 2, x0 + 3.8, y + h / 2,
          colour=TRAIN_EDGE if x0 == 98 else MUTED, lw=2.4 if x0 == 98 else 1.6)
ax.text(100, y + h + 1.6, "patch merging", ha="center", fontsize=8.5, color=TRAIN_EDGE, style="italic")
ax.text(67, y + h + 1.6, "patch merging between stages: grid halves, width doubles", ha="center",
        fontsize=9.5, color=MUTED, style="italic")
ax.text(141, y + h + 1.6, "→ one score per breed", ha="center", fontsize=9.5, color=MUTED)

# ---------------------------------------------------------------- phase bars
def bar(x0, x1, yb, colour, label, sub, label_x=None):
    ax.plot([x0, x1], [yb, yb], color=colour, lw=4, solid_capstyle="butt")
    lx = (x0 + x1) / 2 if label_x is None else label_x
    ax.text(lx, yb - 2.6, label, ha="center", fontsize=11, fontweight="bold", color=colour)
    ax.text(lx, yb - 5.2, sub, ha="center", fontsize=9.3, color=MUTED)

bar(14, 98, 57.5, FROZEN_EDGE, f"Frozen in both phases: {FROZEN_ALWAYS:,} weights",
    "patch embedding, stages 1–3 and their patch merging keep their ImageNet-1K weights")
bar(126, 156, 57.5, HEAD_EDGE, f"Phase 1: {P_HEAD:,} weights train",
    "2 epochs, AdamW lr 3e-4, head only")
bar(98, 156, 47.5, TRAIN_EDGE, f"Phase 2: {PHASE2:,} weights train ({100 * PHASE2 / TOTAL:.0f} %)",
    "3 epochs, AdamW lr 1e-5: last patch merging + stage 4 + final norm + head", label_x=118)
ax.text(4, 47, f"Total {TOTAL:,}\nweights", ha="left", va="center", fontsize=10, color=INK, fontweight="bold")

# legend
for i, (fill, edge, text) in enumerate([(FROZEN_FILL, FROZEN_EDGE, "frozen (pretrained)"),
                                        (TRAIN_FILL, TRAIN_EDGE, "pretrained, fine-tuned in phase 2"),
                                        (HEAD_FILL, HEAD_EDGE, "new, starts random, trains in both phases")]):
    x = 20 + i * 24
    ax.add_patch(Rectangle((x, 45.4), 2.6, 1.8, fc=fill, ec=edge, lw=1.2))
    ax.text(x + 3.4, 46.3, text, va="center", fontsize=9, color=MUTED)

# ---------------------------------------------------------------- bottom left: windows
ax.text(37, 38, "How attention is kept cheap: windows that shift", ha="center", fontsize=13,
        fontweight="bold", color=INK)
gx, gy, gs = 8, 7.5, 23
ax.text(gx + gs / 2, gy + gs + 2.4, "Block A: regular windows", ha="center", fontsize=11, fontweight="bold", color=INK)
for i, (dx, dy) in enumerate([(0, 11.5), (11.5, 11.5), (0, 0), (11.5, 0)]):
    ax.add_patch(Rectangle((gx + dx, gy + dy), 11.5, 11.5, fc=WINDOW_COLOURS[i], ec=INK, lw=1.2))
for k in range(1, 8):
    ax.plot([gx + gs / 8 * k] * 2, [gy, gy + gs], color="white", lw=0.5)
    ax.plot([gx, gx + gs], [gy + gs / 8 * k] * 2, color="white", lw=0.5)
ax.add_patch(Rectangle((gx, gy), gs, gs, fill=False, ec=INK, lw=2))
ax.text(gx + gs / 2, gy - 2.6, "attention only inside each 7 × 7-patch window", ha="center", fontsize=9, color=MUTED)

arrow(33.5, gy + gs / 2, 41, gy + gs / 2)

hx = 43
ax.text(hx + gs / 2, gy + gs + 2.4, "Block B: shifted windows", ha="center", fontsize=11, fontweight="bold", color=INK)
cuts = [0, 6, 17, 23]
c = 0
for i in range(3):
    for j in range(3):
        x0, x1 = cuts[j], cuts[j + 1]
        y0, y1 = cuts[2 - i], cuts[3 - i]
        ax.add_patch(Rectangle((hx + x0, gy + y0), x1 - x0, y1 - y0, fc=WINDOW_COLOURS[c], ec=INK, lw=1.2))
        c += 1
for k in range(1, 8):
    ax.plot([hx + gs / 8 * k] * 2, [gy, gy + gs], color="white", lw=0.5)
    ax.plot([hx, hx + gs], [gy + gs / 8 * k] * 2, color="white", lw=0.5)
ax.add_patch(Rectangle((hx, gy), gs, gs, fill=False, ec=INK, lw=2))
ax.text(hx + gs / 2, gy - 2.6, "windows moved by 3 patches, so neighbours connect", ha="center", fontsize=9, color=MUTED)
ax.text(37, 0.8, "A and B alternate in every stage. Cost grows linearly with image size, not quadratically as in ViT.",
        ha="center", fontsize=9.5, color=INK)

# ---------------------------------------------------------------- bottom right: block pair
ax.text(118, 38, "Inside one pair of Swin blocks", ha="center", fontsize=13, fontweight="bold", color=INK)
for row, (label, attn) in enumerate([("Block A", "Window\nattention"), ("Block B", "Shifted-window\nattention")]):
    yy = 22 - row * 13
    ax.text(84, yy + 3.5, label, ha="center", va="center", fontsize=11, fontweight="bold", color=INK)
    xs = [90, 107, 124, 141]
    for k, (x, text) in enumerate(zip(xs, ["LayerNorm", attn, "LayerNorm", "MLP"])):
        ax.add_patch(FancyBboxPatch((x, yy), 14, 7, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc="#F2F5F8", ec=INK, lw=1.2))
        ax.text(x + 7, yy + 3.5, text, ha="center", va="center", fontsize=9.2, fontweight="bold", color=INK)
        if k < 3:
            arrow(x + 14.2, yy + 3.5, x + 16.8, yy + 3.5)
    arrow(90, yy + 9, 121, yy + 9, colour=HEAD_EDGE, lw=1.3)
    arrow(124, yy + 9, 155, yy + 9, colour=HEAD_EDGE, lw=1.3)
arrow(84, 19.5, 84, 13)
ax.text(122, 5, "Red arrows: skip (residual) connections around each half.", ha="center", fontsize=9.5, color=HEAD_EDGE)
ax.text(122, 2.4, "Each head attends with 32 dimensions; heads double each stage (3, 6, 12, 24).",
        ha="center", fontsize=9.5, color=INK)
ax.text(158, 0.6, "Based on Liu et al. (2021)", ha="right", fontsize=8, color=MUTED, style="italic")

fig.savefig(OUT, facecolor="white")
print(f"saved {OUT}")
