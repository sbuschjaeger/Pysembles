import torch
import torch.nn as nn
import torch.nn.functional as F

from pysembles.Utils import Flatten

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.a1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.a2 = nn.ReLU()

    def forward(self, x):
        out = self.a1(self.bn1(self.conv1(x)))
        out = self.a2(self.bn2(self.conv2(out)))
        out += x
        return nn.functional.max_pool2d(out, kernel_size=2,stride=2)

class SimpleResNet(nn.Module):
    def __init__(self, in_channels = 1, lin_size = None, n_channels = 64, depth = 2, num_classes = 100):
        super().__init__()
        
        if lin_size is None:
            # Try to guess the size of the final linear layer. This probably only works for CIFAR images
            # or similar sizes
            if in_channels == 3:
                if depth == 1:
                    lin_size = 128*n_channels
                elif depth == 2:
                    lin_size = 64*n_channels
                elif depth == 3:
                    lin_size = 16*n_channels
                elif depth == 4:
                    lin_size = 4*n_channels 
                else:
                    lin_size = n_channels
            else:
                if depth == 1:
                    lin_size = 128*n_channels
                elif depth == 2:
                    lin_size = 64*n_channels
                elif depth == 3:
                    lin_size = 16*n_channels
                elif depth == 4:
                    lin_size = 1*n_channels 
                else:
                    lin_size = n_channels
                
        model = [
            nn.Conv2d(in_channels,n_channels,kernel_size=3, padding=1, stride = 1, bias=True),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        ]
        
        for _ in range(depth):
            model.append(
                BasicBlock(n_channels, n_channels)
            )

        model.extend([
            Flatten(),
            # LinearLayer(lin_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            # Activation(),
            nn.Linear(lin_size, num_classes)
        ])

        model = filter(None, model)
        self.layers_ = nn.Sequential(*model)

    def forward(self, x):
        return self.layers_(x)