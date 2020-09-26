#!/usr/bin/env python3

from tqdm import tqdm
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Utils import apply_in_batches, TransformTensorDataset
from .Models import SKEnsemble, SKLearnModel
from .BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear

import copy

class BaggingClassifier(SKEnsemble):
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

    def prepare_backward(self, data, target, weights = None):
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
        self.classes_ = unique_labels(y)
        if self.pipeline:
            X = self.pipeline.fit_transform(X)

        self.X_ = X
        self.y_ = y

        if self.freeze_layers is not None:
            for e in self.estimators_:
                for i, l in enumerate(e.layers_[:self.freeze_layers]):
                    # print("Layer {} which is {} is now frozen".format(i,l))
                    #if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, BinaryConv2d, nn.Linear, BinaryLinear)):
                    for p in l.parameters():
                        p.requires_grad = False
        
        # Check if we use the "fast" method for training. "Fast" copies the entire dataset multiple times and 
        # calls the forward method for each batch manually for each base model. 
        # The other method is the "regular" bagging-style training approach in which we simply call each
        # fit method individually. This trains one model after another which does not fully utilize 
        # the GPU for smaller base models, but might be faster if the base models are already quite large
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
