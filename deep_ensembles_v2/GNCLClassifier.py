#!/usr/bin/env python3
import os

import numpy as np
from tqdm import tqdm

from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable

from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Models import SKEnsemble
from .Utils import apply_in_batches, cov, is_same_func, TransformTensorDataset

class GNCLClassifier(SKEnsemble):
    def __init__(self, n_estimators = 5, l_reg = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.l_reg = l_reg
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])
        # self.l_mode = l_mode

    def prepare_backward(self, data, target, weights = None):
        # TODO Make this use of the weights as well!
        f_bar, base_preds = self.forward_with_base(data)
        
        if isinstance(self.loss_function, nn.MSELoss): 
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]

            eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            D = 2.0*eye_matrix
        elif isinstance(self.loss_function, nn.NLLLoss):
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]
            D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes).type(torch.cuda.FloatTensor)

            eps = 1e-7
            diag_vector = target_one_hot*(1.0/(f_bar**2+eps))
            D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
        elif isinstance(self.loss_function, nn.CrossEntropyLoss):
            n_preds = f_bar.shape[0]
            n_classes = f_bar.shape[1]
            f_bar_softmax = nn.functional.softmax(f_bar,dim=1)
            D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
            diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
            D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
        else:
            # TODO Use autodiff do compute second derivative for given loss function
            # OR Use second formula from paper here? 
            D = torch.tensor(1.0)

        losses = []
        accuracies = []
        diversity = []
        for pred in base_preds:
            diff = pred - f_bar 
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            reg = 1.0/self.n_estimators * 0.5 * covar
            i_loss = self.loss_function(pred, target)
            reg_loss = i_loss - self.l_reg * reg
            
            losses.append(reg_loss)
            accuracies.append(100.0*(pred.argmax(1) == target).type(torch.cuda.FloatTensor))
            diversity.append(reg)

        losses = torch.stack(losses, dim = 1)
        accuracies = torch.stack(accuracies, dim = 1)
        diversity = torch.stack(diversity, dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : losses.sum(dim=1), 
            "metrics" :
            {
                "loss" : self.loss_function(f_bar, target),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(torch.cuda.FloatTensor), 
                "avg loss": losses.mean(dim=1),
                "avg accuracy": accuracies.mean(dim = 1),
                "diversity": diversity.mean(dim = 1)
            } 
            
        }
        return d
