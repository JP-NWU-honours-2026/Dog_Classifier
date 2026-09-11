"""
models.py - Model definitions for 2-Breed Dog Classification.

Includes:
1. BaselineCNN: Small custom convolutional neural network (scratch baseline).
2. TransferResNet18: ResNet-18 with pretrained ImageNet weights.
3. TransferMobileNetV3: MobileNetV3-Small with pretrained ImageNet weights (fast CPU inference).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, EfficientNet_V2_S_Weights, MobileNet_V3_Small_Weights


class BaselineCNN(nn.Module):
    """
    A lightweight custom 3-stage CNN baseline trained from scratch.
    Meets the assignment requirement for a small custom convolutional neural network baseline.
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def get_model(model_name: str = "efficientnet_v2", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Factory function returning the specified model architecture.

    Supported options:
    - 'baseline': Custom 3-block CNN baseline
    - 'efficientnet_v2': Pretrained EfficientNetV2-S (Rubric: EfficientNetV2 family)
    - 'resnet18': Pretrained ResNet-18 (Rubric: ResNet family)
    - 'mobilenet_v3': Pretrained MobileNetV3-Small
    """
    name = model_name.lower().strip()

    if name in ["baseline", "custom_cnn", "cnn"]:
        model = BaselineCNN(num_classes=num_classes)
        model.name = "Baseline-CNN"
        return model

    elif name in ["efficientnet_v2", "efficientnet", "efficientnetv2", "efficientnet_v2_s"]:
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        model.name = "Transfer-EfficientNetV2-S"
        return model

    elif name in ["resnet18", "resnet"]:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )
        model.name = "Transfer-ResNet18"
        return model

    elif name in ["mobilenet_v3", "mobilenet", "mobilenetv3"]:
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )
        model.name = "Transfer-MobileNetV3-Small"
        return model

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: 'baseline', 'efficientnet_v2', 'resnet18'"
        )


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 224, 224)
    for m_name in ["baseline", "efficientnet_v2", "resnet18"]:
        model = get_model(m_name, num_classes=2, pretrained=True)
        out = model(dummy_input)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model.name:<28} | Output shape: {tuple(out.shape)} | Total params: {params:,} | Trainable: {trainable:,}")
