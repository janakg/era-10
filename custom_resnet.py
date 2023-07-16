import torch
from torch import nn
dropout_value_min = 0.03

# Custom Resnet block only. Refer model.py file for the full model
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_value_min)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_value_min)
        )

    def forward(self, x):
        out = self.block1(x)
        out = x + self.block2(out)
        return out