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

from .Models import Ensemble
from .Utils import TransformTensorDataset#, weighted_mse_loss, weighted_squared_hinge_loss

class E2EEnsembleClassifier(Ensemble):
    """ End-to-End (E2E) Learning of the entire ensemble by viewing it as a single model.
    
    Directly E2E training of the entire ensemble. Just get the prediction of each base model, aggregate it and perform SGD on it as if it would be a new fancy Deep Learning architecture. Surprisingly, this approach is often overlooked in literature and sometimes it has strange names. I'll try to gather some references below, but apart from that there is nothing out of the ordinary to explain about this model compared to regular Deep architectures. 

    __References__
        [1] Dutt, A., Pellerin, D., & Quénot, G. (2020). Coupled ensembles of neural networks. Neurocomputing, 396, 346–357. https://doi.org/10.1016/j.neucom.2018.10.092

        [2] Lee, S., Purushwalkam, S., Cogswell, M., Crandall, D., & Batra, D. (2015). Why M Heads are Better than One: Training a Diverse Ensemble of Deep Networks. Retrieved from http://arxiv.org/abs/1511.06314
        
        [3] Webb, A. M., Reynolds, C., Iliescu, D.-A., Reeve, H., Lujan, M., & Brown, G. (2019). Joint Training of Neural Network Ensembles, (4), 1–14. https://doi.org/10.13140/RG.2.2.28091.46880
        
        [4] Webb, A. M., Reynolds, C., Chen, W., Reeve, H., Iliescu, D.-A., Lujan, M., & Brown, G. (2020). To Ensemble or Not Ensemble: When does End-To-End Training Fail? In ECML PKDD 2020 (pp. 1–16). Retrieved from http://www.cs.man.ac.uk/~gbrown/publications/ecml2020webb.pdf
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])

    def prepare_backward(self, data, target, weights = None):
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
            accuracies.append(100.0*(pred.argmax(1) == target).type(self.get_float_type()))

        avg_losses = torch.stack(losses, dim = 1).mean(dim = 1)
        avg_accuracy = torch.stack(accuracies, dim = 1).mean(dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : loss, 
            "metrics" :
            {
                "loss" : loss.detach(),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(self.get_float_type()), 
                "avg loss": avg_losses,
                "avg accuracy": avg_accuracy
            } 
            
        }
        return d
