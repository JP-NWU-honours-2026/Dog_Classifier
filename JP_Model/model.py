"""
model.py - EfficientNetV2 model architecture for 120-Breed Dog Classification.

Uses torchvision's pretrained EfficientNetV2-S with modified classification head.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights


def get_model(
    model_name: str = "efficientnet_v2_s",
    num_classes: int = 120,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Constructs an EfficientNetV2 model adapted for 120 dog breeds.
    """
    name = model_name.lower().strip()

    if name in ["efficientnet_v2_m", "efficientnet_m"]:
        weights = EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_m(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        model.name = "Transfer-EfficientNetV2-M"
        return model

    else:
        # Default: EfficientNetV2-S (fast, balanced, 20M params)
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        model.name = "Transfer-EfficientNetV2-S"
        return model


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    model = get_model(num_classes=120, pretrained=True)
    out = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.name}")
    print(f"Output shape (Batch, Classes): {tuple(out.shape)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
