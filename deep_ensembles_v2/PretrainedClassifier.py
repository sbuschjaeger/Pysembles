#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
import torchvision.models as models

from sklearn.utils.multiclass import unique_labels
from Utils import Flatten
from Models import SKLearnModel

class PretrainedClassifier(SKLearnModel):
    def __init__(self, model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def fit(self, X, y, sample_weight = None):
        if self.model_name == "alexnet":
            model = models.alexnet(pretrained=True)
            layers = list(model.features) + [model.avgpool, Flatten()] + list(model.classifier)
        elif self.model_name == "vgg16":
            model = models.vgg16(pretrained=True)
            layers = list(model.features) + [model.avgpool, Flatten()] + list(model.classifier)
        elif self.model_name == "resnet18":
            model = models.resnet18(pretrained=True)
            layers = [model.conv1, model.bn1, model.relu, model.maxpool] +\
                list(model.layer1) + list(model.layer2) + list(model.layer3) + list(model.layer4) +\
                [model.avgpool, Flatten(), model.fc]
        elif self.model_name == "densenet":
            model = models.densenet121(pretrained=True)
            layers = list(model.features) + [nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(output_size=(1,1)), Flatten()] + list(model.classifier)
        elif self.model_name == "inception":
            model = models.inception_v3(pretrained=True)
            # TODO ADD transform_input if set to true
            layers = [
                model.Conv2d_1a_3x3,
                model.Conv2d_2a_3x3,
                model.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride = 2),
                model.Conv2d_3b_1x1,
                model.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride = 2),
                model.Mixed_5b,
                model.Mixed_5c,
                model.Mixed_5d,
                model.Mixed_6a,
                model.Mixed_6b,
                model.Mixed_6c,
                model.Mixed_6d,
                model.Mixed_6e,
                model.Mixed_7a,
                model.Mixed_7b,
                model.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1,1)),
                nn.Dropout(),
                Flatten(),
                model.fc
            ]
        else:
            pass

        self.layers_ = nn.Sequential(*layers)
        self.classes_ = 1000
        self.X_ = X
        self.y_ = y
        self.cuda()

    def forward(self, X):
        return self.layers_(X)