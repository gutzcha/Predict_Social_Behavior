import torch
import torch.nn as nn
import numpy
from typing import List

config = dict(
    block=dict(filter_sizes=[16,16])



)

class Block(nn.Module):
    def __init__(self, input_dim: int, filter_sizes: List[int] = None, dropout_prob: float = 0.2):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [16, 16]

        block_layers = []
        prev_dim = input_dim
        for dim in filter_sizes:
            block_layers.append(nn.Conv1d(prev_dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=1))
            block_layers.append(nn.BatchNorm1d(dim))
            block_layers.append(nn.ReLU())
            block_layers.append(nn.Dropout(p=dropout_prob))

            prev_dim = dim

        self.layers = nn.Sequential(*block_layers)

    def forward(self, x):
        # apply CONV => BN => RELU x N block to the inputs and return it

        return self.layers(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, layers=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            Block(in_channels, layers)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Block(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)