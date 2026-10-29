"""
model.py - MobileNetV3-Large for 120-Breed Dog Classification.

This file is the ONLY difference between this folder and JP_Model. The data
handling (dataset.py), the training loop (train.py) and the evaluation
(evaluate.py) are copied byte for byte from JP_Model at commit c254375b, so this
model is trained by exactly the same code as the EfficientNetV2-S results it is
compared against.

Why MobileNetV3-Large: it has 4,355,752 trainable parameters with a 120 class
head, against 4,581,784 for the from-scratch CNN in the parent folder. The two
networks are therefore the same size to within five per cent, so the difference
between them isolates the effect of ImageNet pretraining with capacity held
roughly constant. Comparing a 4.6 million parameter scratch model against a 20
or 27 million parameter pretrained one cannot separate those two effects.

The helper functions below are copied unchanged from JP_Model/model.py. They
operate on model.features and model.classifier, which MobileNetV3 also has, so
they work without modification. Only get_model differs.

Weights: MobileNet_V3_Large_Weights.IMAGENET1K_V1 is pinned explicitly rather
than using DEFAULT. DEFAULT resolves to IMAGENET1K_V2, which was trained with an
improved recipe (75.27 against 74.04 per cent ImageNet top-1). The other two
models in this comparison use the standard recipe, so V1 is the matching choice.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes all parameters in the convolutional backbone (model.features),
    keeping only the classifier head trainable.

    Copied unchanged from JP_Model/model.py.
    """
    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = False
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_backbone(model: nn.Module, unfreeze_from_stage: int = 0) -> None:
    """
    Unfreezes backbone layers for fine-tuning.

    Copied unchanged from JP_Model/model.py. The block index means different
    things in the two architectures, so the value has to be chosen rather than
    carried across:

        EfficientNetV2-S   8 feature blocks. JP's --unfreeze_stage 6 leaves
                           74.0 per cent of parameters trainable.
        MobileNetV3-Large 17 feature blocks. Block 13 gives 81.8 per cent and
                           block 14 gives 72.0 per cent.

    **Use --unfreeze_stage 14 here.** Matching the proportion of parameters that
    are adapted is what makes the two runs comparable, since that is what
    determines how much of the pretrained network is actually changed. Matching
    the block number instead would fine-tune noticeably more of this network
    than JP fine-tunes of his.
    """
    if hasattr(model, "features"):
        for i, stage in enumerate(model.features):
            should_train = (i >= unfreeze_from_stage)
            for param in stage.parameters():
                param.requires_grad = should_train

    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_summary(model: nn.Module, verbose: bool = False) -> Dict[str, Any]:
    """
    Returns counts of trainable, frozen, and total parameters.

    Copied unchanged from JP_Model/model.py.
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
    model_name: str = "mobilenet_v3_large",
    num_classes: int = 120,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_initially: bool = False,
) -> nn.Module:
    """
    Constructs MobileNetV3-Large adapted for the breed classification task.

    Architecture pipeline, matching the pattern used for EfficientNetV2-S:
    1. Pretrained convolutional backbone (ImageNet-1K, V1 weights)
    2. Global average pooling, which MobileNetV3 already includes
    3. Classification head ending in Dropout(p=0.3) -> Linear(1280, num_classes)

    One deliberate difference from JP's head. His replaces the whole classifier
    with Dropout then Linear. MobileNetV3's classifier contains a 960 to 1280
    expansion layer that is part of the architecture and carries pretrained
    weights, so that layer is kept and only the final Linear is replaced, with
    the dropout set to 0.3 to match. Removing it would discard pretrained
    parameters and change the network's design rather than just its output size.
    """
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features          # 1280
    model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model.name = "Transfer-MobileNetV3-Large"

    # Attach convenience methods directly to the model instance, as JP's does.
    model.freeze_backbone = lambda: freeze_backbone(model)
    model.unfreeze_backbone = lambda unfreeze_from_stage=0: unfreeze_backbone(model, unfreeze_from_stage)
    model.get_parameter_summary = lambda verbose=False: get_parameter_summary(model, verbose)

    if freeze_initially:
        model.freeze_backbone()

    return model


if __name__ == "__main__":
    print(">>> Testing MobileNetV3-Large transfer learning model (model.py) <<<\n")
    model = get_model(num_classes=120, pretrained=True)

    print("1. Initial model, everything trainable:")
    model.get_parameter_summary(verbose=True)

    print("\n2. Backbone frozen (Phase 1, head only):")
    model.freeze_backbone()
    model.get_parameter_summary(verbose=True)

    print("\n3. Top blocks unfrozen (Phase 2, from block 14 of 17, 72 per cent trainable,")
    print("   matching the 74 per cent that JP's unfreeze_stage 6 leaves trainable):")
    model.unfreeze_backbone(unfreeze_from_stage=14)
    model.get_parameter_summary(verbose=True)

    print("\n4. All blocks unfrozen:")
    model.unfreeze_backbone(unfreeze_from_stage=0)
    model.get_parameter_summary(verbose=True)

    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"\nForward pass successful. Output shape: {tuple(out.shape)}")
