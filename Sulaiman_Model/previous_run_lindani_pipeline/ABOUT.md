# Previous Swin-T run (Lindani-based pipeline), kept for reference

This folder holds the first Swin-T run, from 17–18 October 2026, before the
group moved to JP's pipeline. It is **not** part of the group comparison;
the current runs are in `Sulaiman_Model/results/runs/unclean_transfer/` and
`simplified_transfer/`.

| | This run (`swin_t`) | Current runs |
|---|---|---|
| Pipeline | adapted from Lindani's (`code/`) | JP's (`Sulaiman_Model/*.py`) |
| Split | original 120-class split | JP's original and simplified split files |
| Training | whole network fine-tuned, 15 epochs, AdamW lr 1e-4, warm-up + cosine, weight decay 0.05, no label smoothing, early stopping on val macro F1 | 2 epochs head only (lr 3e-4) + 3 epochs `features[6:]` (lr 1e-5), weight decay 0.01, label smoothing 0.1 |
| Test accuracy / macro F1 | 85.38 % / 0.8490 | 88.03 % / 0.8747 (original split) |

Contents:

- `code/`: `dataset.py`, `model.py`, `train.py`, `evaluate.py` as they were
  for this run. They expect to live directly in `Sulaiman_Model/`, so they
  will not run from this folder. To rerun, check out commit `91131a4c` (or
  earlier) and run them from there.
- `results/runs/swin_t/`: the 15-epoch run (training log, `eval_val/`,
  `eval_test/`). `swin_t_quick/` is the 2-epoch smoke test.
- `swin_notebook.ipynb`, `README.md`, `REPORT_SECTIONS.md`,
  `Swin_T_Report_Sections.docx`: the notebook and documents written for
  this run. Paths inside them refer to the old layout.
