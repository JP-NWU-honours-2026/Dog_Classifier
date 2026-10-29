# Model comparison

Produced by compare_models.py, so every figure in this table is computed
by the same code from each model's own predictions.

| Model | Weights | Parameters | Photos | Accuracy | 95% interval | Top-5 | Macro F1 | Clean accuracy | Training |
|---|---|---|---|---|---|---|---|---|---|
| swin_t | ImageNet-1K pretrained | 27,607,020 | 3164 | 89.92 % | 88.82 to 90.92 | 99.34 % | 0.8953 |  | 9.1 min, NVIDIA GeForce RTX 4060 Laptop GPU |

## Validity check

```
PHOTOGRAPHS SCORED
------------------------------------------------------------------------------
  swin_t                         3164

  All models scored the same 3164 photographs. The comparison is valid.
```
