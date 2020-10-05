#!/usr/bin/env python3

import os
import numpy as np
import torch
import random
import copy

from torch import nn
from torch.autograd import Variable

from tqdm import tqdm

from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

from .Utils import TransformTensorDataset
from .Models import SKLearnModel

class DeepDecisionTreeClassifier(SKLearnModel):
    def __init__(self, split_estimator, leaf_estimator, depth, soft=False, *args, **kwargs):
        super().__init__(base_estimator = lambda: None, *args, **kwargs)
        
        self.depth = depth
        self.split_estimator = split_estimator
        self.leaf_estimator = leaf_estimator
        self.soft = soft

        self.layers_ = []
        self.n_inner = int((2**(self.depth+1) - 1)/2)
        self.n_leafs = 2**self.depth
        for i in range(self.n_inner):
             self.layers_.append(self.split_estimator())
        
        for i in range(self.n_leafs):
             self.layers_.append(self.leaf_estimator())
        self.layers_ = nn.Sequential(*self.layers_)
        
        cur_path = [[0]]
        for i in range(self.depth):
            tmp_path = []
            for p in cur_path:
                p1 = p.copy()
                p2 = p.copy()
                p1.append( 2*p[-1] + 1 )
                p2.append( 2*p[-1] + 2 )
                tmp_path.append(p1)
                tmp_path.append(p2)
            cur_path = tmp_path
        self.all_pathes = cur_path

    def forward(self, x):
        # Execute all models; This can be improved
        all_preds = [l(x) for l in self.layers_]
        path_preds = []
        for path in self.all_pathes:
            # print(path)
            pred = torch.tensor(1.0)
            for i in range(len(path[:-1])):
                cur_node = path[i]
                next_node = path[i+1]
                n_pred = all_preds[cur_node]

                if not self.soft:
                    tmp = n_pred.clone()
                    tmp[tmp >= 0.5] = 1.0
                    tmp[tmp < 0.5] = 0.0
                    n_pred = tmp

                if cur_node == 0:
                    self.cnts[cur_node] += x.shape[0]
                else:
                    self.cnts[cur_node] += (pred != 0).sum().item()
                
                if (next_node % 2) == 0:
                    pred = n_pred * pred
                else:
                    pred = (1.0 - n_pred) * pred
            
            self.cnts[path[-1]] += (pred != 0).sum().item()
            pred = pred * all_preds[path[-1]]
            path_preds.append(pred)
        # asdf
        tmp = torch.stack(path_preds)
        
        return tmp.sum(dim = 0)
