"""
model.py

The baseline CNN: a small convolutional network trained from scratch with no
pretrained weights. It is the control in the three way comparison against JP's
EfficientNetV2-S and Sulaiman's Swin-T, both of which start from ImageNet
weights.

The architecture and the reasoning behind every choice are recorded in
MODEL_DESIGN.md. In short: five blocks, each halving the image and doubling the
number of pattern detectors, then a global average pooling head.

    224 -> 112 -> 56 -> 28 -> 14 -> 7 pixels
    3   -> 32  -> 64 -> 128 -> 256 -> 512 channels

Run this file on its own to print the architecture, the exact trainable
parameter count and a shape test:

    python Lindani_Model/model.py

Needs: pip install torch
"""

import torch
import torch.nn as nn

INPUT_SIZE = 224        # same input size as the other two models
NUM_CLASSES = 120       # one output per breed
CHANNELS = (32, 64, 128, 256, 512)
DROPOUT = 0.3


def conv_block(in_channels, out_channels, convs=1):
    """
    One block of the network. With convs=1 it is four steps; with convs=2 the
    convolution, normalisation and activation repeat before the single pooling
    step, which deepens the block without changing the image size ladder.

    The four steps:

      Conv2d      slides 3 by 3 windows over the image looking for patterns.
                  out_channels is how many different patterns it looks for.
                  padding=1 keeps the image the same size, so only the pooling
                  step changes it. bias=False because the batch norm that
                  follows applies its own shift, making a bias redundant.
      BatchNorm2d rescales the output so the next layer receives numbers in a
                  sensible range. Without it a network trained from scratch is
                  slow and fussy about the learning rate.
      ReLU        keeps positive responses and zeroes negative ones, which is
                  what lets the network learn something other than a straight
                  line. inplace=True saves memory.
      MaxPool2d   halves the width and height, keeping the strongest response in
                  each 2 by 2 patch, so the next block sees a larger area of the
                  original photograph through the same size window.
    """
    layers = []
    for n in range(convs):
        layers += [
            nn.Conv2d(in_channels if n == 0 else out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class BaselineCNN(nn.Module):
    """
    The baseline network. Five convolutional blocks, then a head that turns the
    final feature maps into 120 breed scores.
    """

    def __init__(self, num_classes=NUM_CLASSES, channels=CHANNELS, dropout=DROPOUT,
                 convs_per_block=None):
        super().__init__()

        # One convolution per block is iteration 1, the design in MODEL_DESIGN.md.
        # Passing (1, 1, 1, 2, 2) doubles the convolutions in the last two blocks,
        # which is the deeper variant tested in iteration 2.
        convs_per_block = convs_per_block or (1,) * len(channels)
        assert len(convs_per_block) == len(channels), \
            "convs_per_block must give one number per block"
        self.convs_per_block = tuple(convs_per_block)

        blocks = []
        in_channels = 3  # a colour photograph starts with red, green and blue
        for out_channels, convs in zip(channels, self.convs_per_block):
            blocks.append(conv_block(in_channels, out_channels, convs))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)

        # Global average pooling: each of the 512 final channels is averaged down
        # to a single number, giving a 512 number summary of the photograph. The
        # alternative, flattening 512 x 7 x 7 into a fully connected layer, would
        # add hundreds of thousands of parameters, and with roughly 120 training
        # photographs per breed those are spent memorising. See MODEL_DESIGN.md.
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),        # ignores some features while training,
                                          # so the model cannot lean on any one
            nn.Linear(in_channels, num_classes),
        )

        self._initialise_weights()

    def _initialise_weights(self):
        """
        Sets the starting values of every weight.

        This matters more here than for the other two models. They start from
        ImageNet weights; this one starts from numbers we choose. He
        initialisation scales the random starting values by the size of each
        layer, which keeps the signal from shrinking or exploding as it passes
        through a deep ReLU network.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """One photograph in, 120 breed scores out."""
        return self.classifier(self.pool(self.features(x)))


def count_parameters(model):
    """The number the report must quote: how many values training adjusts."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    model = BaselineCNN()
    print(model)

    print("\nVARIANTS")
    for label, convs in [("iteration 1, one conv per block", (1, 1, 1, 1, 1)),
                         ("iteration 2, two convs in blocks 4 and 5", (1, 1, 1, 2, 2))]:
        print(f"  {label:<44} {count_parameters(BaselineCNN(convs_per_block=convs)):>10,}")

    print("\nTRAINABLE PARAMETERS")
    for n, block in enumerate(model.features, 1):
        print(f"  block {n}          {count_parameters(block):>10,}")
    print(f"  head             {count_parameters(model.classifier):>10,}")
    print(f"  total            {count_parameters(model):>10,}")

    # A shape test. Two photographs are pushed through so the batch dimension is
    # exercised, and the size after every block is printed. The numbers should
    # read 112, 56, 28, 14, 7, matching MODEL_DESIGN.md.
    print("\nSHAPE THROUGH THE NETWORK, for a batch of 2 photographs")
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    print(f"  input            {tuple(x.shape)}")
    model.eval()
    with torch.no_grad():
        for n, block in enumerate(model.features, 1):
            x = block(x)
            print(f"  after block {n}    {tuple(x.shape)}")
        x = model.pool(x)
        print(f"  after pooling    {tuple(x.shape)}")
        scores = model.classifier(x)
        print(f"  output           {tuple(scores.shape)}")

    assert scores.shape == (2, NUM_CLASSES), "the head must produce one score per breed"
    print(f"\nOutput is 2 photographs by {NUM_CLASSES} breed scores, as required.")
    print(f"Chance level on {NUM_CLASSES} classes is {100 / NUM_CLASSES:.2f} per cent.")


if __name__ == "__main__":
    main()
