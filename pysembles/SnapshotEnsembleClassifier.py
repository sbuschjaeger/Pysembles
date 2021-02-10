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

from .Models import Ensemble
from .Utils import apply_in_batches, cov, TransformTensorDataset#, weighted_mse_loss, weighted_squared_hinge_loss

class SnapshotEnsembleClassifier(Ensemble):
    """ SnapshotEnsembleClassifier.

    Snapshot ensembles are ensembles which take snapshots of the current model during optimization and stores it. During prediction _all_ models provide a prediction and the average is returned. This way, a diverse ensemble is built on-the-fly while training a single model [1,2]. Snapshot ensembles formally belong to the class of pseudo-ensembles [3]. Pseudo-ensembles are ensembles that are derived from a large single network by perturbing it with a noise process. They minimize the following objective

    $$
    \\frac{1}{N}\sum_{j=1}^N \mathbb E_{\\theta} [ \ell_{\\theta}(\mu(x_i),y_i) ] + \lambda \mathbb E_{\\theta} [Z(\mu(x_i), \mu_{\\theta}(x_i))]
    $$

    

    where \(\mu\) denotes the `mother' net, \( \mu_{\\theta} \) is a child net under the noise process \(\\theta, \ell\) is a loss function and \(Z\) is a regularizer with regularization strength \(\lambda\). As an interestring sidenote, Dropout also belongs to this class. 

    This implementation accepts a list of epochs (starting by 0) and takes a snapshot of the current model at the beginning of the provided epoch. For example, if you supply the 0 you will store the random initialization of the model, before any training happend. If you train your model for a shorter period than provided, then no snapshots are taken. 

    Attributes:
        list_of_snapshots (list of int): A list of epochs at which the snapshots should be taken. Snapshots are taken at the beginning of each epoch.

    __References__
    
    [1] Hu, H., Sun, W., Venkatraman, A., Hebert, M., & Bagnell, J. A. (2017). Gradient Boosting on Stochastic Data Streams, 54. Retrieved from https://arxiv.org/pdf/1703.00377.pdf

    [2] Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger, K. Q. (2017). Snapshot ensembles: Train 1, get M for free. 5th International Conference on Learning Representations, ICLR 2017 - Conference Track Proceedings, 1–14. Retrieved from https://arxiv.org/pdf/1704.00109.pdf

    [3] Bachman, P., Alsharif, O., & Precup, D. (2014). Learning with pseudo-ensembles. In Advances in Neural Information Processing Systems.
    """
    def __init__(self, list_of_snapshots, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_of_snapshots = list_of_snapshots
        self.estimators_ = nn.ModuleList([ self.base_estimator() ])

    def restore_state(self, checkpoint):
        super().restore_state(checkpoint)
        self.list_of_snapshots = checkpoint["list_of_snapshots"]

    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "list_of_snapshots":self.list_of_snapshots,
        }

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
            accuracies.append(100.0*(pred.argmax(1) == target).type(self.get_float_type()))

        avg_losses = torch.stack(losses, dim = 1).mean(dim = 1)
        avg_accuracy = torch.stack(accuracies, dim = 1).mean(dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : loss, 
            "metrics" :
            {
                "loss" : loss,
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(self.get_float_type()), 
                "avg loss": avg_losses,
                "avg accuracy": avg_accuracy
            } 
            
        }
        return d
