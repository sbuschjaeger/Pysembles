import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

from deep_ensembles_v2.Utils import Flatten

class VGGNet(nn.Module):
    def __init__(self, input_size = (3,32,32), n_channels = 32, depth = 4, hidden_size = 512, p_dropout = 0.0, n_classes = 100):
        super().__init__()

        in_channels = input_size[0]
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

        # This is kinda hacking but it works well to automatically determine the size of the linear layer
        x = torch.rand((1,*input_size)).type(torch.FloatTensor)
        for l in model:
            x = l(x)
        lin_size = x.view(1,-1).size()[1]
        # in_channels = 1?
        # if n_layers == 1:
        #     lin_size = 506*n_channels #
        # elif n_layers == 2:
        #     lin_size = 242*n_channels
        # elif n_layers == 3:
        #     lin_size = 75*n_channels
        # elif n_layers == 4:
        #     lin_size = 16*n_channels
        # else:
        #     lin_size = 5*n_channels
        # in_channels = 3?
        # if depth == 1:
        #     lin_size = 128*n_channels
        # elif depth == 2:
        #     lin_size = 64*n_channels
        # elif depth == 3:
        #     lin_size = 48*n_channels
        # else:
        #     lin_size = 16*n_channels

        model.extend(
            [
                Flatten(),
                nn.Linear(lin_size, hidden_size),
                nn.Dropout(p_dropout) if p_dropout > 0 else None,
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_classes)
            ]
        )

        model = filter(None, model)
        self.layers_ = nn.Sequential(*model)

    def forward(self, x):
        return self.layers_(x)
    