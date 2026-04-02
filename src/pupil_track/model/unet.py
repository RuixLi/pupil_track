"""Shallow U-Net for pupil segmentation.

3 max-pool levels, 128x128 grayscale input, 1-channel sigmoid output.
~1.9M parameters (vs ~31M for the standard 4-level U-Net).
"""

import torch
import torch.nn as nn

from .parts import DoubleConv, Down, Up, OutConv


class ShallowUNet(nn.Module):
    """Shallow U-Net: 3 encoder levels, 128x128 input, binary output.

    Architecture:
        inc:   1 -> 32   (128x128)
        down1: 32 -> 64  (64x64)
        down2: 64 -> 128 (32x32)
        down3: 128 -> 256 (16x16)  [bottleneck]
        up1:   256+128 -> 128  (32x32)
        up2:   128+64 -> 64   (64x64)
        up3:   64+32 -> 32    (128x128)
        outc:  32 -> 1        (128x128)
    """

    def __init__(self, n_channels: int = 1, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.outc = torch.utils.checkpoint(self.outc)
