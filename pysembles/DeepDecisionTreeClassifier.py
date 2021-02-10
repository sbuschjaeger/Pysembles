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
from .Models import Model

class DeepDecisionTreeClassifier(Model):
    '''
    A soft decision tree classifier implementation via Neural Networks. *NOTE*: This implementation is not really maintained at the moment. 
    The main idea is to train a soft dt classifier in an end-to-end fashion. To do so, the DT architecture must be given beforehand. This implementation uses balanced trees of depth \( d \). On each level there are 2^d nodes and each node has 2 children except for leaf nodes. The formal prediction function is

    $$
    f(x) = \sum_{l \in L} g_l(x) \prod_{i \in \mathcal P(l)} s_i(x)
    $$
    where \( L \) is the set of leaves, \( \mathcal P(l) \) denotes the path from the root node to the leaf node \( l \), \( s_i \) are the corresponding prediction functions for each split and \( g_l \) is the prediction function of the leaf. Formally we define these functions as:

    - \( s_i \colon \mathcal X \mapsto [0,1] \), where \( \mathcal X \) is the domain of the input data. Conceptually, \(s_i(x) \\rightarrow 0 \) means that the examples belongs to the left child, whereas \(s_i(x) \\rightarrow 1 \) means it belongs to the right child. Note that regular DTs use axis-aligned splits with \(s_i(x) \in \\{0,1\\} \). 
    - \( g_l \colon \mathcal X \mapsto \mathbb R^C \), where \(\mathcal X\) is the domain of the input data and \( C \) is the number of classes

    Attributes:
        split_estimator (function): A function that returns a new split estimator. Make sure that the output of the newly created estimators is between 0 and 1 (e.g. by using a nn.Sigmoid).

        leaf_estimator (function): A function that returns a new leaf estimator.

        depth (int): The maximum depth of this DT. A depth of 0 means that we only 1 leaf estimator is trained.

        soft (bool, optional): True if the original split values between 0 and 1 should be retained. If false, these values are mapped ot 0 or 1, via \( 1\{s_i(x) \ge 0.5 \}\). Defaults to False.


    __References__:
        TODO

    '''
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

    def restore_state(self,checkpoint):
        super().restore_state(checkpoint)

        self.depth = checkpoint["depth"]
        self.split_estimator = checkpoint["split_estimator"]
        self.leaf_estimator = checkpoint["leaf_estimator"]
        self.soft = checkpoint["soft"]
        self.n_inner = checkpoint["n_inner"]
        self.n_leafs = checkpoint["n_leafs"]
        self.all_pathes = checkpoint["all_pathes"]

        for i in range(self.n_inner):
             self.layers_.append(self.split_estimator())
        
        for i in range(self.n_leafs):
             self.layers_.append(self.leaf_estimator())
        
        self.layers_ = nn.Sequential(*self.layers_)
        self.layers_.load_state_dict(checkpoint["layers_state_dict"])

    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "depth":self.depth,
            "split_estimator":self.split_estimator,
            "leaf_estimator":self.leaf_estimator,
            "soft":self.soft,
            "n_inner":self.n_inner,
            "n_leafs":self.n_leafs,
            "all_pathes":self.all_pathes,
            "layers_state_dict":self.layers_.state_dict()
        } 

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
