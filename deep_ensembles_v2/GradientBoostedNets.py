#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

from torch import nn
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Models import SKEnsemble 

class GradientBoostedNets(SKEnsemble):
    def __init__(self, n_estimators = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])
        
        # Currently we only support average for this kind of training I guess. 
        # Not sure how to change the overall training procedure to fit this tbh.
        assert self.combination_type == "average"

    def prepare_backward(self, data, target, weights = None):
        f_bar, base_preds = self.forward_with_base(data)

        staged_pred = None
        total_loss = None

        for pred in base_preds:
            if staged_pred is None:
                staged_pred = pred
            else:
                staged_pred = staged_pred.detach() + pred

            if weights is not None:
                loss = self.loss_function(staged_pred, target) * weights
            else:
                loss = self.loss_function(staged_pred, target) 

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        d = {
            "prediction" : f_bar, 
            "backward" : total_loss, 
            "metrics" :
            {
                "loss" : self.loss_function(f_bar, target),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(torch.cuda.FloatTensor), 
            } 
            
        }
        return d
