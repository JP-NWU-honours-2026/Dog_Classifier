"""
model.py  (Sulaiman_Model, Swin Transformer)

Step 2: the model itself. Swin-T (Swin Transformer Tiny), pretrained on
ImageNet-1K, with its last layer swapped for one that outputs 120 dog breeds.

How transfer learning works here, in two parts:

  1. The BACKBONE (everything except the last layer) is kept exactly as
     torchvision ships it, with weights learned on 1.28 million ImageNet
     photos. It already knows what fur, ears, eyes and legs look like.
     It turns a 224 x 224 photo into a summary of 768 numbers.

  2. The HEAD (the last layer) is thrown away and replaced. The original
     head maps those 768 numbers to ImageNet's 1,000 classes. Ours maps them
     to our 120 breeds, and starts from random numbers.

During training (step 3) the whole network is fine-tuned: the new head learns
from scratch and the backbone is nudged gently, using a small learning rate so
the pretrained knowledge isn't wiped out.

How Swin differs from JP's and Lindani's CNNs, for the report:
  - The photo is cut into 4 x 4 pixel patches, and each patch becomes a vector.
  - Self-attention lets each patch compare itself with other patches. To keep
    this cheap, patches only attend within small windows (7 x 7 patches).
  - Every second layer, the windows SHIFT by half a window, so information can
    pass between neighbouring windows. This is the "Shifted Window" in the name.
  - Patches are merged between stages (56 -> 28 -> 14 -> 7 patches across),
    building a hierarchy similar to a CNN's, which is what makes Swin practical
    for normal image sizes.

Self-test, from the repository root:

    python Sulaiman_Model/model.py

Measured parameter counts (from the self-test, 17 October 2026):
    total            27,611,634   all trainable (full fine-tuning)
    new head             92,280   (768 x 120 weights + 120 biases)
    backbone         27,519,354   pretrained on ImageNet-1K
Compare: Lindani's baseline CNN has 1,631,128, about 17 times fewer.

The first run downloads the pretrained weights (about 110 MB) into PyTorch's
cache folder. Later runs reuse them.
"""

import torch
import torch.nn as nn
from torchvision.models import Swin_T_Weights, swin_t

NUM_CLASSES = 120
DROPOUT = 0.3
WEIGHTS = Swin_T_Weights.IMAGENET1K_V1   # named explicitly so the report can state it


def build_model(num_classes=NUM_CLASSES, dropout=DROPOUT, pretrained=True):
    """
    Returns Swin-T with a new 120-breed head.

    pretrained=False gives the same architecture with random weights. It is
    only there for testing without the download, never for the real run.
    """
    model = swin_t(weights=WEIGHTS if pretrained else None)

    in_features = model.head.in_features            # 768 for Swin-T
    model.head = nn.Sequential(
        nn.Dropout(p=dropout),                      # randomly blanks 30% of the 768 numbers in training
        nn.Linear(in_features, num_classes),        # 768 -> 120
    )
    return model


def count_parameters(model, trainable_only=True):
    """Number of weights in the model. Trainable = the ones training will update."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def main():
    print("SELF-TEST\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    head = model.head[1]
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model)
    head_params = count_parameters(model.head)

    print(f"  model                    Swin-T (swin_t), torchvision")
    print(f"  pretrained weights       {WEIGHTS}")
    print(f"  new head                 Dropout({DROPOUT}) -> Linear({head.in_features} -> {head.out_features})")
    print(f"  total parameters         {total:,}")
    print(f"  trainable parameters     {trainable:,}")
    print(f"  of which the new head    {head_params:,}")
    print(f"  backbone (pretrained)    {total - head_params:,}")

    # Push two fake photos through, to prove the shapes line up end to end.
    model.eval()
    dummy = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  device                   {device}")
    print(f"  input shape              {tuple(dummy.shape)}   (2 photos, 3 colours, 224 x 224)")
    print(f"  output shape             {tuple(out.shape)}   (2 photos, one score per breed)")
    assert out.shape == (2, NUM_CLASSES), "output does not have one score per breed"

    if device == "cuda":
        print(f"  GPU memory in use        {torch.cuda.memory_allocated() / 1024**2:.0f} MB")

    print("\nSelf-test passed.")


if __name__ == "__main__":
    main()
