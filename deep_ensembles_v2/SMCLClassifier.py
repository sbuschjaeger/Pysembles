#!/usr/bin/env python3
import warnings

import torch
from torch import nn

from .Models import SKEnsemble


class SMCLClassifier(SKEnsemble):
    """ Stochastic Multiple Choice Learning Classifier.

    Attributes:
        n_estimators (int): Number of estimators in ensemble

    References:
        Lee, S., Purushwalkam, S., Cogswell, M., Ranjan, V., Crandall, D., & Batra, D. (2016). Stochastic multiple choice learning for training diverse deep ensembles. Advances in Neural Information Processing Systems, 1(Nips), 2127–2135. Retrieved from http://papers.nips.cc/paper/6270-stochastic-multiple-choice-learning-for-training-diverse-deep-ensembles.pdf
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])

        if self.combination_type is not None and self.combination_type != "best":
            warnings.warn("SMCL is usually evaluated on the oracle loss, which means the _best_ predictor across the ensemble is used. You explicitly set it to '{}' which does not make sense. I am going to fix that for you now".format(self.combination_type)) 
        self.combination_type = "best"

    def prepare_backward(self, data, target, weights = None):
        f_bar, base_preds = self.forward_with_base(data)

        losses = []
        accuracies = []
        for i, pred in enumerate(base_preds):
            if weights is None:
                iloss = self.loss_function(pred, target)
            else:
                # TODO: PyTorch copies the weight vector if we use weights[:,i] to index
                #       a specific row. Maybe we should re-factor this?
                iloss = self.loss_function(pred, target) * weights[:,i].cuda()

            losses.append(iloss)
            accuracies.append(100.0*(pred.argmax(1) == target).type(torch.cuda.FloatTensor))

        losses = torch.stack(losses, dim = 1)
        lmin, _ = losses.min(dim=1)
        accuracies = torch.stack(accuracies, dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : lmin, 
            "metrics" :
            {
                "loss" : self.loss_function(f_bar, target),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(torch.cuda.FloatTensor), 
                "avg loss": losses.mean(dim=1),
                "avg accuracy": accuracies.mean(dim = 1),
            } 
            
        }
        return d
