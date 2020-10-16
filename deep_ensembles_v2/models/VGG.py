import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

from deep_ensembles_v2.Utils import Flatten

class VGGNet(nn.Module):
    # TODO Make this more generic with hidden_size / n_channels / depth
    def __init__(self, in_channels = 3, model_size = "large"):
        super().__init__()

        if "large" in model_size:
            hidden_size = 512
            n_channels = 32
            depth = 4
        else:
            hidden_size = 256
            n_channels = 16
            depth = 3

        def make_layers(level, n_channels):
            return [
                nn.Conv2d(in_channels if level == 0 else level*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
                nn.BatchNorm2d((level+1)*n_channels),
                nn.ReLU(),
                nn.Conv2d((level+1)*n_channels, (level+1)*n_channels, kernel_size=3, padding=1, stride = 1, bias=True),
                nn.BatchNorm2d((level+1)*n_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,stride=2)
            ]

        model = []
        for i in range(depth):
            model.extend(make_layers(i, n_channels))

        if depth == 1:
            lin_size = 128*n_channels
        elif depth == 2:
            lin_size = 64*n_channels
        elif depth == 3:
            lin_size = 48*n_channels
        else:
            lin_size = 16*n_channels

        model.extend(
            [
                Flatten(),
                nn.Linear(lin_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 100)
            ]
        )

        model = filter(None, model)
        self.layers_ = nn.Sequential(*model)

    def forward(self, x):
        return self.layers_(x)
    