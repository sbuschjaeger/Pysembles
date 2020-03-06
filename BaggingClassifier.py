#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn

from sklearn.utils.multiclass import unique_labels

from Models import SKLearnModel
from Models import StagedEnsemble
from BinarisedNeuralNetworks import BinaryConv2d

import copy

class BaggingClassifier(StagedEnsemble):
    def __init__(self, n_estimators = 5, bootstrap = True, frac_examples = 1.0, freeze_layers = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.frac_samples = frac_examples
        self.bootstrap = bootstrap
        self.freeze_layers = freeze_layers
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, y): 
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        self.estimators_ = nn.ModuleList([
            SKLearnModel(training_csv="training_{}.csv".format(i), *self.args, **self.kwargs) for i in range(self.n_estimators)
        ])
        
        if self.freeze_layers:
            for e in self.estimators_:
                for l in e.layers_[:-1]:
                    # 
                    l.requires_grad = False
                    # if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, BinaryConv2d)):
                    #     l.requires_grad = False

        for idx, est in enumerate(self.estimators_):
            if self.seed is not None:
                np.random.seed(self.seed + idx)

            idx_array = [i for i in range(len(y))]
            idx_sampled = np.random.choice(
                idx_array, 
                size=int(self.frac_samples*len(idx_array)), 
                replace=self.bootstrap
            )

            X_sampled = X[idx_sampled,] 
            y_sampled = y[idx_sampled]
            est.fit(X_sampled, y_sampled)
