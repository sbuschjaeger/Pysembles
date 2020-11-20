#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Models import Model, Ensemble

class StackingClassifier(Model, Ensemble):
    """ Stacking Classifier.

    Stacking stacks the predictions of each base learner into one large vector and then trains another model on this new
    "example" vector. This implementation can be viewed as End2End stacking, in which both - the base models as well as
    the combinator model - are trained in an end-to-end fashion. 

    Attributes:
        classifier (function): Generates and returns a new classifier. Please make sure, that it accepts the stacked input. 
            The dimension of the input will likely change with different n_estimators. 

    References:
        - Wolpert, D. (1992). Stacked Generalization ( Stacking ). Neural Networks.
        - Breiman, L. (1996). Stacked regressions. Machine Learning. https://doi.org/10.1007/bf00117832
    """
    def __init__(self, classifier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])
        self.classifier = classifier
        self.classifier_ = self.classifier()

    def restore_state(self,checkpoint):
        super().restore_state(checkpoint)
        # I am not sure if model is part of the overall state_dict and thus properly loaded. 
        # Lets go the save route and store it as well
        self.classifier = checkpoint["classifier"]
        self.classifier_ = self.classifier()
        self.classifier_.load_state_dict(checkpoint["classifier_state_dict"])

    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "classifier":self.classifier,
            "classifier_state_dict":self.classifier_.state_dict(),
        }  

    def forward(self, x):
        base_preds = [est(x) for est in self.estimators_]
        stacked_pred = torch.flatten(torch.stack(base_preds, dim=1), start_dim=1)
        return self.classifier_(stacked_pred)
