#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Models import SKLearnModel
# from Models import Flatten
from .BinarisedNeuralNetworks import BinaryTanh
from .BinarisedNeuralNetworks import BinaryLinear

class StackingClassifier(SKLearnModel):
    def __init__(self, n_estimators, classifier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])
        self.classifier = classifier
        self.classifier_ = self.classifier()

    def forward(self, x):
        base_preds = [est(x) for est in self.estimators_]
        stacked_pred = torch.flatten(torch.stack(base_preds, dim=1), start_dim=1)
        return self.classifier_(stacked_pred)
