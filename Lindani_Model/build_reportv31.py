"""
build_reportv31.py

Produces ReportV3.1.docx from ReportV1.docx, applying the group's review points:

  1. Section numbering matched to the specification's required nine sections,
     including every cross-reference in the text
  2. A sentence interpreting what the F1 scores in the weakest-classes table mean
  3. More interpretation of the learning curves
  4. A shorter problem statement
  5. A title page with group members, and a table of contents
  7. Two figures documenting the duplicate detection

It also clears the corrections still outstanding from the V3 fix list: the
Figure 1 caption, the duplicated Figure 1 line, the in-text table numbers that
sat one ahead of the captions, the three untitled table captions, a typo in the
Table 6 caption, and the label-audit limitation line.

ReportV1.docx is read only and never modified.

    python Lindani_Model/build_reportv31.py
"""

import re
import shutil

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from split_check import LINDANI_DIR

SOURCE = LINDANI_DIR / "ReportV1.docx"
OUTPUT = LINDANI_DIR / "ReportV3.1.docx"
DUPES = LINDANI_DIR / "results" / "duplicate_check"

# The specification puts the title, members and abstract in section 1, so every
# section from the dataset onwards shifts up by one.
SECTION_SHIFT = {7: 8, 6: 7, 5: 6, 4: 5, 3: 4, 2: 3}

# Two cleaning figures go into section 3.4, so the figures after them move down.
FIGURE_SHIFT = {6: 8, 5: 7, 4: 6, 3: 5}

TITLE = ("Fine-Grained Dog Breed Classification: Comparing Transfer Learning "
         "Against a Network Trained From Scratch")

MEMBERS = [
    ("JP Retief", "38279118"),
    ("Lindani Mdlalose", "[STUDENT NUMBER]"),
    ("Sulaiman Sulaman", "[STUDENT NUMBER]"),
]

HEADINGS = {
    "abstract": "1. Abstract",
    "the problem": "2. Introduction and problem motivation",
    "dataset": "3. Dataset, classes, preparation and ethics",
    "model selection and methodology": "4. Methodology, models and transfer learning",
    "experimental setup and reproducibility": "5. Experimental setup and hyperparameters",
    "results": "6. Results, figures and model comparison",
    "discussion": "7. Discussion, limitations and error analysis",
    "conclusion": "8. Conclusion",
    "references": "9. References",
}


def find(doc, needle):
    for para in doc.paragraphs:
        if needle in para.text:
            return para
    return None


def set_text(para, text):
    """Replaces a paragraph's text, keeping the style of its first run."""
    if para.runs:
        para.runs[0].text = text
        for run in para.runs[1:]:
            run.text = ""
    else:
        para.add_run(text)


def swap(para, old, new, report):
    if old in para.text:
        set_text(para, para.text.replace(old, new))
        report.append(f"{old!r} -> {new!r}")
        return True
    return False


def delete(para, report, why):
    para._element.getparent().remove(para._element)
    report.append(f"deleted: {why}")


def insert_after(anchor, doc, text="", style=None):
    para = doc.add_paragraph(text, style=style)
    anchor._p.addnext(para._p)
    return para


def insert_figure(anchor, doc, image, caption, report, width=5.9):
    if not image.is_file():
        report.append(f"MISSING FIGURE {image.name}")
        return insert_after(anchor, doc, f"[MISSING FIGURE: {image.name}]")
    picture = insert_after(anchor, doc)
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(image), width=Inches(width))
    label = insert_after(picture, doc, caption, style="Caption")
    label.alignment = WD_ALIGN_PARAGRAPH.CENTER
    report.append(f"figure added: {caption[:40]}")
    return label


def shift_numbers(doc, shift, kind, report):
    """
    Shifts numbering up, highest first so a number already changed is not
    changed twice. Only clear references are matched, so values in the prose
    such as "1.70 to 1" and "2.5 per cent" are left alone.
    """
    patterns = []
    for old in sorted(shift, reverse=True):
        new = shift[old]
        if kind == "section":
            patterns += [
                (re.compile(rf"\bSection {old}\.(\d)"), rf"Section {new}.\1"),
                (re.compile(rf"\bSection {old}([:,)])"), rf"Section {new}\1"),
                (re.compile(rf"^Section {old}\.(\d)"), rf"Section {new}.\1"),
                (re.compile(rf"^{old}\.(\d)\b"), rf"{new}.\1"),
                (re.compile(rf"^{old}\s+([A-Z])"), rf"{new} \1"),
            ]
        else:
            patterns.append((re.compile(rf"\bFigure {old}\b"), f"Figure {new}"))

    changed = 0
    for para in doc.paragraphs:
        before = para.text
        after = before
        for pattern, replacement in patterns:
            after = pattern.sub(replacement, after)
        if after != before:
            set_text(para, after)
            changed += 1
    report.append(f"{kind} numbers shifted in {changed} paragraphs")


def tidy_headings(doc, report):
    """
    Numbers the top-level headings as the specification does, rewrites the three
    subsection headings that read "Section 2.4, Cleaning" into the "3.4 Cleaning"
    form used by the rest, and gives every numbered subsection the same style so
    it appears in the table of contents.
    """
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if para.style.name.startswith("Heading"):
            plain = re.sub(r"^(Section\s*)?\d+[.,:]?\s*", "", text).strip().lower()
            for key, new in HEADINGS.items():
                if plain.startswith(key):
                    set_text(para, new)
                    para.style = doc.styles["Heading 1"]
                    report.append(f"heading: {new}")
                    break
            continue

        # "Section 3.4, Cleaning" -> "3.4 Cleaning"
        match = re.match(r"^Section (\d\.\d)[.,:]?\s+(.*)$", text)
        if match:
            set_text(para, f"{match.group(1)} {match.group(2)}")
            para.style = doc.styles["Heading 2"]
            report.append(f"subheading normalised: {para.text}")
            continue

        # Any other numbered subsection heading, however it was styled.
        if re.match(r"^\d\.\d\s+\S", text) and len(text) < 70:
            if para.style.name != "Heading 2":
                para.style = doc.styles["Heading 2"]
                report.append(f"subheading restyled: {text}")


def add_front_matter(doc, report):
    """A title page and a table of contents, placed before the abstract."""
    first = doc.paragraphs[0]
    blocks = []

    def line(text, size=None, bold=False, space_before=0):
        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_before = Pt(space_before)
        run = para.add_run(text)
        run.bold = bold
        if size:
            run.font.size = Pt(size)
        blocks.append(para)
        return para

    line("", space_before=0)
    line(TITLE, size=22, bold=True, space_before=90)
    line("ITRI626 Deep Learning for Image Classification", size=13,
         space_before=18)
    line("North-West University", size=13)
    line("Group members", size=12, bold=True, space_before=48)
    for name, number in MEMBERS:
        line(f"{name}    {number}", size=12)
    line("November 2026", size=12, space_before=36)

    page_break = doc.add_paragraph()
    page_break.add_run().add_break(WD_BREAK.PAGE)
    blocks.append(page_break)

    contents = doc.add_paragraph("Table of contents", style="Heading 1")
    blocks.append(contents)

    # A real table of contents field. Word fills it in and gets the page numbers
    # right for whatever the final layout turns out to be.
    field = doc.add_paragraph()
    run = field.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = r'TOC \o "1-2" \h \z \u'
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = ("Right-click this line in Word and choose Update Field "
                        "to build the contents.")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, placeholder, end):
        run._r.append(element)
    blocks.append(field)

    after = doc.add_paragraph()
    after.add_run().add_break(WD_BREAK.PAGE)
    blocks.append(after)

    for block in blocks:
        first._p.addprevious(block._p)
    report.append("title page, group members and table of contents added")


def main():
    shutil.copy(SOURCE, OUTPUT)
    doc = Document(OUTPUT)
    report = []

    # ---- 4. a shorter problem statement ---------------------------------
    # The whole 255-word opening is a single paragraph in the source, so it is
    # replaced in one piece.
    para = find(doc, "Image classification is a task of assigning an image")
    if para:
        set_text(para,
            "Fine-grained image classification assigns an image to one of several classes "
            "that all belong to the same broader category. It is difficult because the "
            "differences between classes are small and confined to certain parts of the "
            "image, while photographs of the same class vary widely in pose, lighting, scale "
            "and background. Dog breeds show this clearly: the Eskimo dog and the Siberian "
            "husky share coat pattern and build, and Stanford Dogs was constructed as a "
            "benchmark for exactly this kind of distinction (Khosla et al., 2011). The "
            "problem is also of practical relevance, since breed information is used by "
            "animal shelters, veterinary practices and pet owners, and is often recorded "
            "inaccurately when assigned by eye.")
        report.append("problem statement shortened (first paragraph, 255 to 95 words)")

        second = insert_after(para, doc,
            "Deep networks need large volumes of labelled data to learn reliable features, "
            "and fine-grained datasets rarely provide it; Stanford Dogs offers roughly 170 "
            "photographs per class. Transfer learning addresses this by initialising a "
            "network with weights learned on a large dataset such as ImageNet (Deng et al., "
            "2009) and adapting them to the new task. On Stanford Dogs that advantage is "
            "difficult to interpret, because the dataset was itself drawn from ImageNet, so "
            "pretrained models may have encountered some of its photographs during "
            "pretraining.")
        second.style = para.style
        report.append("problem statement split into two paragraphs")

    # ---- 3. interpret the learning curves -------------------------------
    para = find(doc, "The learning curves for all models are shown")
    if para:
        insert_after(para, doc,
            "The two shapes differ in a way that reflects where each model starts. The "
            "pretrained models are already above 70 per cent validation accuracy after a "
            "single epoch, because their features are useful before training begins and only "
            "the new classification layer has to be learned; the remaining epochs add a few "
            "points as the upper layers adapt. The custom CNN starts near chance and climbs "
            "slowly, since it has to learn edges, textures and shapes before it can learn "
            "breeds at all, and it needs 59 epochs to reach a figure the pretrained models "
            "pass in one. Its training and validation curves also separate from about epoch "
            "20, with training loss still falling while validation loss flattens, which is "
            "the overfitting described in Section 4.5. The pretrained models show no such "
            "separation in five epochs, so their curves had not levelled off and their "
            "scores are best read as lower bounds."
        ).style = para.style
        report.append("learning curve interpretation added")

    # ---- 2. interpret the F1 scores -------------------------------------
    para = find(doc, "lists the weakest classes for the best and")
    if para:
        insert_after(para, doc,
            "F1 combines two things into one number between 0 and 1: precision, the share of "
            "photographs called a breed that really were that breed, and recall, the share of "
            "that breed the model actually found. A breed scoring 0.20 is therefore one the "
            "model gets wrong far more often than right, in both directions, while the "
            "strongest breeds in this project score above 0.90. Macro F1 is the plain average "
            "across all 120 breeds, so unlike accuracy it gives a rare breed the same weight "
            "as a common one, which is why it sits close to but below each model's accuracy. "
            "The weakest breeds are largely the same ones for every model, which points at "
            "the data rather than at any single architecture."
        ).style = para.style
        report.append("F1 interpretation added")

    # ---- figure numbers make room for the two cleaning figures ----------
    shift_numbers(doc, FIGURE_SHIFT, "figure", report)

    # ---- 7. cleaning evidence -------------------------------------------
    anchor = find(doc, "Cleaning began with an integrity check")
    if anchor:
        label = insert_figure(
            anchor, doc, DUPES / "pairs_confirmed_crossing.png",
            "Figure 3: Confirmed duplicate pairs that fall in different partitions. "
            "A red caption marks a photograph in the test set whose copy is in the "
            "training set.", report)
        insert_figure(
            label, doc, DUPES / "pairs_distance_0.png",
            "Figure 4: Pairs at a difference-hash distance of zero. Several carry two "
            "different breed labels, which is how the contradictory labels were found.",
            report)

    # ---- the Figure 1 caption, and its duplicate ------------------------
    duplicate = find(doc, "Figure 1: Representative photographs from each of the 120")
    if duplicate and duplicate.style.name != "Caption":
        delete(duplicate, report, "duplicated Figure 1 line above the image")
    caption = find(doc, "Figure 1: Representative photographs from twelve")
    if caption:
        set_text(caption,
                 "Figure 1: Representative photographs from each of the 120 "
                 "Stanford Dogs classes")
        report.append("Figure 1 caption: twelve -> 120")

    # ---- 1. section numbering -------------------------------------------
    shift_numbers(doc, SECTION_SHIFT, "section", report)

    para = find(doc, "The report proceeds from the data to the models")
    if para:
        set_text(para,
            "The report proceeds from the data to the models: Section 3 documents the "
            "dataset and its preparation, Section 4 describes the model families and the "
            "transfer learning strategy, and Section 5 sets out the experimental "
            "configuration. The results are presented in Section 6 and interpreted in "
            "Section 7, which also addresses classification errors and limitations, before "
            "Section 8 concludes with the recommended model.")
        report.append("roadmap paragraph renumbered")

    tidy_headings(doc, report)

    # ---- table numbers in the text, which sat one ahead of the captions --
    fixes = [
        ("Three model families were compared", "(Table 4)", "(Table 3)"),
        ("lists the training settings", "Table 5 lists", "Table 4 lists"),
        ("final configuration was chosen from five runs", "(Table 7)", "(Table 6)"),
        ("Swin-T achieved the highest accuracy", "Table 8", "Table 7"),
        ("lists the weakest classes for the best and", "Table 9 lists", "Table 8 lists"),
        ("Removing the 34 test photographs", "(Table 8)", "(Table 7)"),
    ]
    for anchor_text, old, new in fixes:
        para = find(doc, anchor_text)
        if para:
            swap(para, old, new, report)

    # ---- captions that were left without a title -------------------------
    titles = [
        ("Table 1", "Table 1: Dataset composition and image properties"),
        ("Table 3", "Table 3: The model families compared"),
        ("Table 7", "Table 7: Test set performance of all four models"),
        ("Table 8", "Table 8: Weakest classes by F1 for three of the models"),
    ]
    for bare, titled in titles:
        for para in doc.paragraphs:
            if para.style.name == "Caption" and para.text.strip() == bare:
                set_text(para, titled)
                report.append(f"caption titled: {titled}")
                break

    para = find(doc, "Custome CNN dev runs")
    if para:
        set_text(para, "Table 6: Custom CNN development runs")
        report.append("Table 6 caption typo fixed")

    para = find(doc, "Figure 5")
    for para in doc.paragraphs:
        if para.style.name == "Caption" and para.text.strip() == "Figure 7":
            set_text(para, "Figure 7: Swin-T confusion matrix, with the most "
                           "confused breeds enlarged")
            report.append("caption titled: Figure 7")
            break

    # ---- the label-audit limitation he dropped on purpose ----------------
    para = find(doc, "The label audit covered only 100 photographs")
    if para:
        delete(para, report, "label-audit limitation line")

    # ---- 5. front matter -------------------------------------------------
    add_front_matter(doc, report)

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}\n")
    for line in report:
        print("  -", line)


if __name__ == "__main__":
    main()
