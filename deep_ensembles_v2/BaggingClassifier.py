#!/usr/bin/env python3

from tqdm import tqdm
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Utils import TransformTensorDataset
from .Models import SKEnsemble, SKLearnModel

import copy

class BaggingClassifier(SKEnsemble):
    """ Classic Bagging in the modern world of Deep Learning. 

    Bagging uses different subsets of features / training points to train an ensemble of classifiers. The classic version of Bagging uses bootstrap samples which means that each base learner roughly receives 63% of the training data, whereas roughly 37% of the training data are duplicates. This lets each base model slightly overfit to their respective portion of the training data leading to a somewhat diverse ensemble. 

    This implementation supports a few variations of bagging. Similar to SKLearn you can choose the fraction of samples with and without bootstrapping. Moreover, you can freeze all but the last layer of each base model. This simulates a form of feature sampling / feature extraction, and should be expanded in the future. Last, there is a "fast" training method which jointly trains the ensemble using poisson weights for each individual classifier. 

    Attributes:
        n_estimators (int): Number of estimators in ensemble. Should be at least 1
        bootstrap (bool): If true, sampling is performed with replacement. If false, sampling is performed without replacement
        frac_examples (float): Fraction of training examples used per base learner, that is N_base = (int) N * self.frac_examples if N is the number of training data points. 
            Must be from (0,1].
        freeze_layers (bool): If true, all but the last layer of all base learners are frozen and _not_ fitted during training (requires_grad = False is set to false). 
            This may simulate something similar to feature bagging
        train_method (str): If set to "fast" a (arguably) faster training method is used. 
            "Fast" implements an online version of Bagging, which weights each example by sampling values from a Poisson distribution as proposed by Oza et al. in 2001. A similar approach called Wagging has also been evaluated by Webb in 2000 in the context (batch) decision tree learning.
            This online approach to Bagging can be faster for smaller base models which do not utilize the entire GPU. The reason for this is, that CUDA calls are evaluated asynchronous leading to a better overall utilization of the GPU if multiple. Moreover, we can directly monitor the overall ensemble loss which is nice. 
            The other method (anything where train_method != "fast") is the "regular" bagging-style training approach in which we simply call each fit method individually. This trains one model after another which might be faster if the base models are already quite large and fully utilize the GPU. NOTE: The fast method currently does not support frac_examples and ignores this parameter

    References:
    - Breiman, L. (1996). Bagging predictors. Machine Learning. https://doi.org/10.1007/bf00058655
    - Webb, G. I. (2000). MultiBoosting: a technique for combining boosting and wagging. Machine Learning. https://doi.org/10.1023/A:1007659514849
    - Oza, N. C., & Russell, S. (2001). Online Bagging and Boosting. Retrieved from https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf 
    """
    def __init__(self, n_estimators = 5, bootstrap = True, frac_examples = 1.0, freeze_layers = None, train_method = "fast", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.frac_samples = frac_examples
        self.bootstrap = bootstrap
        self.freeze_layers = freeze_layers
        self.train_method = train_method
        self.estimators_ = nn.ModuleList([
            SKLearnModel(training_file="training_{}.jsonl".format(i), *args, **kwargs) for i in range(self.n_estimators)
        ])

        assert self.frac_samples > 0 and self.frac_samples <= 1.0, "frac_examples expects the fraction of samples used, this must be between (0,1]. It was {}".format(self.frac_samples)

    def prepare_backward(self, data, target, weights = None):
        # TODO WHAT ABOUT frac_samples?
        f_bar, base_preds = self.forward_with_base(data)

        accuracies = []
        losses = []
        for i, pred in enumerate(base_preds):
            # During training we set the weights to a poisson distribution (see below).
            # However, during testing this function might also be executed. In this case, we 
            # dont want to weight models, but use all of them equally for computing statistics.
            if weights is None:
                iloss = self.loss_function(pred, target)
            else:
            # TODO: PyTorch copies the weight vector if we use weights[:,i] to index
            #       a specific row. Maybe we should re-factor this?
                iloss = self.loss_function(pred, target) * weights[:,i].cuda() 

            losses.append(iloss)
            accuracies.append(100.0*(pred.argmax(1) == target).type(torch.cuda.FloatTensor))

        losses = torch.stack(losses, dim = 1)
        accuracies = torch.stack(accuracies, dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : losses.sum(dim=1), 
            "metrics" :
            {
                "loss" : self.loss_function(f_bar, target),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(torch.cuda.FloatTensor), 
                "avg loss": losses.mean(dim=1),
                "avg accuracy": accuracies.mean(dim = 1)
            } 
            
        }
        return d

    def fit(self, X, y): 
        # TODO: INLCUDE SAMPLE_WEIGHTS!!!

        self.classes_ = unique_labels(y)
        if self.pipeline:
            X = self.pipeline.fit_transform(X)

        self.X_ = X
        self.y_ = y

        # TODO REWORK THIS. IT CURRENTLY ASSUMES THAT EACH BASE LEARNER HAS layers_ WHICH MIGHT NOT BE THE CASE
        if self.freeze_layers is not None:
            for e in self.estimators_:
                for i, l in enumerate(e.layers_[:self.freeze_layers]):
                    for p in l.parameters():
                        p.requires_grad = False
        
        # Check if we use the "fast" method for training. 
        if self.train_method != "fast":
            for idx, est in enumerate(self.estimators_):
                if self.seed is not None:
                    np.random.seed(self.seed + idx)

                idx_array = [i for i in range(len(y))]
                idx_sampled = np.random.choice(
                    idx_array, 
                    size=int(self.frac_samples*len(idx_array)), 
                    replace=self.bootstrap
                )
                X_sampled = X[idx_sampled,] 
                y_sampled = y[idx_sampled]
                est.fit(X_sampled, y_sampled)
        else:
            # We follow the idea of Oza and Russell in "Online Bagging and Boosting" (https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
            # which suggest to weight each example with w ~ Possion(1) for each base-learner
            w_tensor = torch.poisson(torch.ones(size=(len(y), self.n_estimators))).numpy()
            super().fit(X,y,w_tensor)
