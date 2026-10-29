# Model comparison

Produced by compare_models.py, so every figure in this table is computed
by the same code from each model's own predictions.

| Model | Weights | Parameters | Photos | Accuracy | 95% interval | Top-5 | Macro F1 | Clean accuracy | Training |
|---|---|---|---|---|---|---|---|---|---|
| baseline_cnn | none (from scratch) | 4,581,784 | 3200 | 53.84 % | 52.11 to 55.57 | 84.78 % | 0.532 | 53.73 % | 64.4 min, Tesla T4 |
| swin_t | ImageNet-1K pretrained | 27,611,634 | 3200 | 88.03 % | 86.86 to 89.11 | 99.31 % | 0.8747 | 88.21 % | 9.1 min, NVIDIA GeForce RTX 4060 Laptop GPU |

## Validity check

```
PHOTOGRAPHS SCORED
------------------------------------------------------------------------------
  baseline_cnn                   3200
  swin_t                         3200

  All models scored the same 3200 photographs. The comparison is valid.
```
