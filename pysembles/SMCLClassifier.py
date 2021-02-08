#!/usr/bin/env python3
import warnings

import torch
from torch import nn

from .Models import Ensemble


class SMCLClassifier(Ensemble):
    """ Stochastic Multiple Choice Learning Classifier.

    As often argued, diversity might be important for ensembles to work well. Stochastic Multiple Choice Learning (SMCL)
    enforces diversity by training each expert model on a subset of the training data for which it already works
    pretty well. Due to the random initialization each ensemble member is likely to perform better or worse on different
    parts of the data and thereby introducing diversity. SMCL enforces this specialization by selecting the best
    expert (wrt. to the loss) for each example and then only updates that one expert for that example. All other experts
    will never receive that example. 

    __References__
        [1] Lee, S., Purushwalkam, S., Cogswell, M., Ranjan, V., Crandall, D., & Batra, D. (2016). Stochastic multiple choice learning for training diverse deep ensembles. Advances in Neural Information Processing Systems, 1(Nips), 2127–2135. Retrieved from http://papers.nips.cc/paper/6270-stochastic-multiple-choice-learning-for-training-diverse-deep-ensembles.pdf
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
            accuracies.append(100.0*(pred.argmax(1) == target).type(self.get_float_type()))

        losses = torch.stack(losses, dim = 1)
        lmin, _ = losses.min(dim=1)
        accuracies = torch.stack(accuracies, dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : lmin, 
            "metrics" :
            {
                "loss" : self.loss_function(f_bar, target),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(self.get_float_type()), 
                "avg loss": losses.mean(dim=1),
                "avg accuracy": accuracies.mean(dim = 1),
            } 
            
        }
        return d
