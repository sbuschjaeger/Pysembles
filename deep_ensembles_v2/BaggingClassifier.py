#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn

from sklearn.utils.multiclass import unique_labels

from Models import SKLearnModel
from Models import StagedEnsemble
from BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear

import copy

class RandomFeature(nn.Module):
    def __init__(self, dimension, prob_one = 0.75):
        super().__init__()
        # low = 0 (inclusive), high = 1 (exclusive) -> use 2 here
        #self.random_projection = torch.randint(0,2,dimension).cuda()
        
        self.prob_one = prob_one
        self.random_projection = torch.zeros(size=dimension).cuda()
        self.random_projection.bernoulli_(self.prob_one)
        self.random_projection.requires_grad = False

    def forward(self, x):
        # TODO REMOVE squeeze ?
        # TODO MAKE SURE SIZE IS CORRECT?
        x_tmp = x.squeeze(1)
        rnd = torch.cat(x.shape[0]*[self.random_projection.squeeze(1)])
        #print("rnd: ", rnd.shape)
        tmp2 = torch.bmm(x_tmp,rnd.transpose(1,2))
        tmp2 = tmp2.unsqueeze(1)
        #print(x.shape)
        #print(tmp2.shape)
        return tmp2

        #rnd = self.random_projection.squeeze(1)
        # print(self.random_projection.shape)
        # print(x.shape)
        #print("X:", x_tmp.shape)
        #print("RND:", rnd.transpose(1,2).shape)
        #tmp2 = torch.bmm(x_tmp,rnd.view(rnd.shape[1],rnd.shape[2],rnd.shape[0]))

        # tmp = x * self.random_projection
        # print("TMP: ", tmp)
        # print("TMP2: ", tmp2)
        # print(tmp.shape)
        # print(tmp2.shape)
        # asdf
        # return tmp

class BaggingClassifier(StagedEnsemble):
    def __init__(self, n_estimators = 5, bootstrap = True, frac_examples = 1.0, freeze_layers = None, random_features = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.frac_samples = frac_examples
        self.bootstrap = bootstrap
        self.freeze_layers = freeze_layers
        self.args = args
        self.kwargs = kwargs
        self.random_features = random_features

    def fit(self, X, y): 
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        self.estimators_ = nn.ModuleList([
            SKLearnModel(training_csv="training_{}.csv".format(i), *self.args, **self.kwargs) for i in range(self.n_estimators)
        ])
        
        if self.freeze_layers is not None:
            for e in self.estimators_:
                for i, l in enumerate(e.layers_[:self.freeze_layers]):
                    # print("Layer {} which is {} is now frozen".format(i,l))
                    #if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, BinaryConv2d, nn.Linear, BinaryLinear)):
                    for p in l.parameters():
                        p.requires_grad = False
        
        if self.random_features:
            for e in self.estimators_:
                # TODO Make sure that if there is a transformer in the base estimator, that the shape is also correct here
                combined = [RandomFeature(dimension = X[0].shape), *e.layers_]
                e.layers_ = nn.Sequential( *combined)
                # print(e.layers_)
                # asdf

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
