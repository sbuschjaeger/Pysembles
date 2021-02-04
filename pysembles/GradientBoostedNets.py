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

from .Models import Ensemble 

class GradientBoostedNets(Ensemble):
    ''' (Stochastic) Gradient Boosting for Neural Networks
    Gradient Boosting sequentially trains a classifier on the residuals of the ensemble. In its most basic form this is usually a stagewise process in which a new classifier is trained in each stage / round [1]. Related Boosting algorithms like AdaBoost can also be viewed in this framework [2,3]. In order to speed-up the training process, Stochastic Gradient boosting [4] has been proposed. Stochastic Gradient Boosting trains the individual classifiers in each round on a random sample of the entire dataset. This implementation is a variation of this process and combines it with Stochastic Gradient Descent. The basic idea is to freeze the weights of individual models \( h^1,\dots, h^{i-1} \) and perform SGD on the i-th model's parameter using \( \ell(\\frac{1}{M}\sum_{j=1}^i h^j(x), y) \). This process is repeated for each model.

    As far as I know this specific training has not been discussed in literature so far, but there is some significant overlap with exsiting work. See e.g. [5] and references therein. 

    __References__

    [1] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Retrieved from https://www.jstor.org/stable/pdf/2699986.pdf?seq=1
    
    [2] Mason, L., Baxter, J., Bartlet, P., & Frean, M. (1999). Boosting Algorithms as Gradient Descent in Function Space. Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.51.6893&rep=rep1&type=pdf

    [3] Schapire, R. E., & Freund, Y. (2012). Boosting: Foundations and algorithms. MIT press.

    [4] Friedman, J. H. (1999). Stochastic Gradient Boosting, 1(3), 1–10. Retrieved from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.1666&rep=rep1&type=pdf

    [5] Hu, H., Sun, W., Venkatraman, A., Hebert, M., & Bagnell, J. A. (2017). Gradient Boosting on Stochastic Data Streams, 54. Retrieved from https://arxiv.org/pdf/1703.00377.pdf

    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(self.get_float_type()), 
            } 
            
        }
        return d
