"""
model.py - Swin-T (Swin Transformer Tiny) for Dog Breed Classification.

Sulaiman_Model copy of JP_Model/model.py with the same functions and the
same two-phase pattern, using Swin-T instead of EfficientNetV2.

Swin-T's model.features holds 8 parts:
  0 patch embedding   1 stage 1   2 patch merging   3 stage 2
  4 patch merging     5 stage 3   6 patch merging   7 stage 4
unfreeze_from_stage=6 (JP's default) trains the last merge and stage 4, plus
the final norm and the head - the equivalent of JP's top EfficientNet stages.

Implements standard Transfer Learning patterns:
1. Feature Extraction (Phase 1): Backbone frozen, only classifier head trains.
2. Fine-Tuning (Phase 2): Backbone layers unfrozen (full or top stages) with reduced LR.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all parameters in the convolutional backbone (model.features),
    keeping only the classifier head trainable.

    Equivalent to:
        base_model.trainable = False
    in TensorFlow / Keras.
    """
    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: nn.Module, unfreeze_from_stage: int = 0) -> None:
    """
    Unfreezes backbone layers for fine-tuning.

    Parameters:
        unfreeze_from_stage: Starting stage index in model.features (0-7 for Swin-T).
                             If 0, unfreezes all stages.
                             If > 0 (e.g. 5 or 6), freezes stages < unfreeze_from_stage
                             and unfreezes stages >= unfreeze_from_stage.

    Equivalent to:
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
    in TensorFlow / Keras.
    """
    if hasattr(model, "features"):
        for i, stage in enumerate(model.features):
            should_train = (i >= unfreeze_from_stage)
            for param in stage.parameters():
                param.requires_grad = should_train

    for param in list(model.norm.parameters()) + list(model.head.parameters()):
        param.requires_grad = True


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_summary(model: nn.Module, verbose: bool = False) -> Dict[str, Any]:
    """
    Returns counts of trainable, frozen, and total parameters,
    similar to Keras model.summary().
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_pct": round(trainable_pct, 2),
    }

    if verbose:
        print(f"--------------------------------------------------")
        print(f" Model Parameter Summary: {getattr(model, 'name', 'Model')}")
        print(f"--------------------------------------------------")
        print(f" Total Parameters:     {total_params:,}")
        print(f" Trainable Parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
        print(f" Frozen Parameters:    {frozen_params:,} ({100.0 - trainable_pct:.2f}%)")
        print(f"--------------------------------------------------")

    return summary


def get_model(
    model_name: str = "swin_t",
    num_classes: int = 120,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_initially: bool = False,
) -> nn.Module:
    """
    Constructs Swin-T adapted for the dog breeds.

    Architecture pipeline:
    1. Pretrained Swin-T backbone (ImageNet-1K, Swin_T_Weights.IMAGENET1K_V1)
    2. Final LayerNorm + global average pooling (part of torchvision's swin_t)
    3. Custom classification head: Dropout(p=0.3) -> Linear(768, num_classes)
    """
    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_t(weights=weights)
    in_features = model.head.in_features  # 768
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    model.name = "Transfer-Swin-T"

    # Attach convenience methods directly to the model instance
    model.freeze_backbone = lambda: freeze_backbone(model)
    model.unfreeze_backbone = lambda unfreeze_from_stage=0: unfreeze_backbone(model, unfreeze_from_stage)
    model.get_parameter_summary = lambda verbose=False: get_parameter_summary(model, verbose)

    if freeze_initially:
        model.freeze_backbone()

    return model


if __name__ == "__main__":
    print(">>> Testing Transfer Learning Model Pattern (model.py) <<<\n")
    model = get_model(num_classes=120, pretrained=True)

    print("1. Initial Model (Default all trainable):")
    model.get_parameter_summary(verbose=True)

    print("\n2. Freezing Backbone (Phase 1: Feature Extraction):")
    model.freeze_backbone()
    model.get_parameter_summary(verbose=True)

    print("\n3. Unfreezing Top Stages (Phase 2: Fine-Tuning features[6:] = last merge + stage 4):")
    model.unfreeze_backbone(unfreeze_from_stage=6)
    model.get_parameter_summary(verbose=True)

    print("\n4. Unfreezing All Stages (Phase 2: Full Fine-Tuning):")
    model.unfreeze_backbone(unfreeze_from_stage=0)
    model.get_parameter_summary(verbose=True)

    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"\nForward pass successful. Output shape: {tuple(out.shape)}")
