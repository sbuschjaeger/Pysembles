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
from .Utils import apply_in_batches, cov, TransformTensorDataset#, weighted_mse_loss, weighted_squared_hinge_loss

class LazryEnsembleClassifier(SKEnsemble):
    def __init__(self, n_estimators = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimators_ = nn.ModuleList() 

    def prepare_backward(self, data, target, weights = None):
        
        # Randomly copy a model 

        f_bar, base_preds = self.forward_with_base(data)
        loss = self.loss_function(f_bar, target)

        if weights is not None:
            loss = loss * weights

        accuracies = []
        losses = []
        for pred in base_preds:
            iloss = self.loss_function(pred, target)
            if weights is not None:
                iloss *= weights 
            losses.append(iloss.detach())
            accuracies.append(100.0*(pred.argmax(1) == target).type(torch.cuda.FloatTensor))

        avg_losses = torch.stack(losses, dim = 1).mean(dim = 1)
        avg_accuracy = torch.stack(accuracies, dim = 1).mean(dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : loss, 
            "metrics" :
            {
                "loss" : loss.detach(),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(torch.cuda.FloatTensor), 
                "avg loss": avg_losses,
                "avg accuracy": avg_accuracy
            } 
            
        }
        return d
