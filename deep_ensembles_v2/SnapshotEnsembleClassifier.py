import os
import copy

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

class SnapshotEnsembleClassifier(SKEnsemble):
    def __init__(self, list_of_snapshots, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_of_snapshots = list_of_snapshots
        self.estimators_ = nn.ModuleList([ self.base_estimator() ])

    def prepare_backward(self, data, target, weights = None):
        if self.cur_epoch in self.list_of_snapshots and self.batch_cnt == 0 and len(self.estimators_) < self.n_estimators:
            self.estimators_.append(copy.deepcopy(self.estimators_[-1]))

            for param in self.estimators_[-2].parameters():
                param.requires_grad = False

            # It would be more efficient to remove the current parameter group from the optimizer, as discussed
            # here https://discuss.pytorch.org/t/delete-parameter-group-from-optimizer/46814
            # However, AdaBelief has no param_group which we used in our experiments and there is no mention of this
            # in the PyTorch documentation. So for now we just set the gradients to false. 
            #del self.optimizer.param_group[0] # optim.param_group = []
            self.optimizer.add_param_group({'params' : self.estimators_[-1].parameters()})
            #print("Created a new estimators, now we have {}".format(len(self.estimators_)))

        # We set requires_grad to false in all other models except the last one. Thus we can call "forward_with_base"
        # as usual which gives the base_preds for taking statistics etc. 
        # If we use f_bar to compute the loss and the backward step, then only the last model should be updated
        f_bar, base_preds = self.forward_with_base(data)
        loss = self.loss_function(f_bar, target)

        if weights is not None:
            loss = loss * weights

        accuracies = []
        losses = []
        
        for i, pred in enumerate(base_preds):
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
                "loss" : loss,
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(torch.cuda.FloatTensor), 
                "avg loss": avg_losses,
                "avg accuracy": avg_accuracy
            } 
            
        }
        return d
