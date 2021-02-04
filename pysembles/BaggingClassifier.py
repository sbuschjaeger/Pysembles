#!/usr/bin/env python3

import numpy as np
import torch

from torch import nn
from torch.utils.data import Sampler
from torch.utils.data import Dataset

from .Utils import TransformTensorDataset
from .Models import Ensemble, Model

import copy

class BootstrapSampler(Sampler):
    """ Implements bootstrap sampling in PyTorch
    Attributes:
        N (int): The total number of training datasets
        seed (long): The random seed
        bootstrap (bool): If true, sample with replacement else sample without replacement
        frac_examples (float): Fraction of training examples used for sampling N_sampled = (int) N * self.frac_examples. Must be from (0,1]. 
    """
    def __init__(self, N, seed = 12345, bootstrap = True, frac_examples = 1.0):
        self.bootstrap = bootstrap
        self.frac_examples = frac_examples
        assert self.frac_samples > 0 and self.frac_samples <= 1.0, "frac_examples expects the fraction of samples used, this must be between (0,1]. It was {}".format(self.frac_samples)

        np.random.seed(seed)
        idx_array = [i for i in range(N)]
        self.idx_sampled = np.random.choice(
            idx_array, 
            size=int(self.frac_examples*len(idx_array)), 
            replace=self.bootstrap
        )

    def __iter__(self):
        return iter(self.idx_sampled)

    def __len__(self):
        return len(self.idx_sampled)

class WeightedDataset(Dataset):
    """ A weighted dataset in PyTorch. Each example / target pair in the dataset receives a pre-computed weight. The dataset and the weight tensor should have the same length len(dataset) == len(w_tensor)
    
    Attributes:
        dataset: The original dataset
        w_tensor: A tensor of weights.
    """
    def __init__(self, dataset, w_tensor):
        self.dataset = dataset
        self.w_tensor = w_tensor
        assert len(dataset) == len(w_tensor), "Dataset and w_tensor should have the same size in WeightedDataset but received len(dataset) = {} and len(w_tensor) = {}".format(len(dataset, len(w_tensor)))

    def __getitem__(self, index):
        #items = self.dataset.__getitem__(index)
        items = self.dataset[index]
        weights = self.w_tensor[index]

        return (*items, weights)

    def __len__(self):
        return len(self.dataset)

class BaggingClassifier(Ensemble):
    """ Classic Bagging in the modern world of Deep Learning. 

    Bagging uses different subsets of features / training points to train an ensemble of classifiers [1]. The classic version of Bagging uses bootstrap samples which means that each base learner roughly receives 63% of the training data, whereas roughly 37% of the training data are duplicates. This lets each base model slightly overfit to their respective portion of the training data leading to a somewhat diverse ensemble. 

    This implementation supports a few variations of bagging. Similar to SKLearn you can choose the fraction of samples with and without bootstrapping. There is also a "fast" training method which jointly trains the ensemble using Poisson weights for each individual classifier. 

    Attributes:
        n_estimators (int): Number of estimators in the ensemble. Should be at least 1
        bootstrap (bool): If true, sampling is performed with replacement. If false, sampling is performed without replacement. Only has an effect if train_method = "bagging"
        frac_examples (float): Fraction of training examples used per base learner \( N\_base = (int) N * frac\_examples \) if N is the number of training data points. Must be from (0,1]. Only has an effect if train_method = "bagging"
        train_method (str): There are 3 modes:

            - `bagging`: The "regular" bagging-style training approach in which we compute bootstrap samples and train each estimators individually on its respective sample as presented in [1]. This trains one model after another which might be faster if the base models are already quite large and fully utilize the GPU. The size and type of sample can be controlled via `frac_examples` and `bootstrap` parameter. Please note, that currently this mode cannot be properly restored from a checkpoint.

            - `wagging`: Computes continuous Poisson weights which are used during fit as presented in [2]. This method can be faster for smaller base models which do not utilize the entire GPU. Moreover, we can follow the entire ensemble loss during optimization. It is important to note, that the Poisson weight is applied to the loss. The `frac_examples` and `bootstrap` parameters are ignored here.

            - `fast bagging` (or any other string which is not {`bagging`, `wagging`}). Computes discrete Poisson weights which are used during fit as presented in [3]. This method can be faster for smaller base models which do not utilize the entire GPU. Moreover, we can follow the entire ensemble loss during optimization. It is important to note, that the Poisson weight is applied to the loss. The `frac_examples` and `bootstrap` parameters are ignored here.

    __References:__

    [1] Breiman, L. (1996). Bagging predictors. Machine Learning. https://doi.org/10.1007/bf00058655

    [2] Webb, G. I. (2000). MultiBoosting: a technique for combining boosting and wagging. Machine Learning. https://doi.org/10.1023/A:1007659514849

    [3] Oza, N. C., & Russell, S. (2001). Online Bagging and Boosting. Retrieved from https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf 
    """
    def __init__(self, bootstrap = True, frac_examples = 1.0, train_method = "fast bagging", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frac_samples = frac_examples
        self.bootstrap = bootstrap
        self.train_method = train_method
        self.args = args
        self.kwargs = kwargs

        assert self.frac_samples > 0 and self.frac_samples <= 1.0, "frac_examples expects the fraction of samples used, this must be between (0,1]. It was {}".format(self.frac_samples)

    def restore_state(self, checkpoint):
        super().restore_state(checkpoint)
        self.bootstrap = checkpoint["bootstrap"]
        self.train_method = checkpoint["train_method"]

    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "frac_samples":self.frac_samples,
            "train_method":self.train_method
        } 

    def prepare_backward(self, data, target, weights = None):
        f_bar, base_preds = self.forward_with_base(data)

        accuracies = []
        losses = []
        for i, pred in enumerate(base_preds):
            # During training we set the weights to a Poisson distribution (see below).
            # However, during testing this function might also be executed. In this case, we 
            # dont want to weight models, but use all of them equally for computing statistics.
            if weights is None:
                iloss = self.loss_function(pred, target)
            else:
                # TODO: PyTorch copies the weight vector if we use weights[:,i] to index
                #       a specific row. Maybe we should re-factor this?
                #tmp = self.loss_function(pred, target)
                iloss = self.loss_function(pred, target) * weights[:,i].type(self.get_float_type())

            losses.append(iloss)
            accuracies.append(100.0*(pred.argmax(1) == target).type(self.get_float_type()))

        losses = torch.stack(losses, dim = 1)
        accuracies = torch.stack(accuracies, dim = 1)

        d = {
            "prediction" : f_bar, 
            "backward" : losses.sum(dim=1), 
            "metrics" :
            {
                "loss" : self.loss_function(f_bar, target),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(self.get_float_type()), 
                "avg loss": losses.mean(dim=1),
                "avg accuracy": accuracies.mean(dim = 1)
            } 
            
        }
        return d

    def fit(self, data):
        if self.train_method == "bagging":
            self.estimators_ = nn.ModuleList()

            # TODO Offer proper support for store / load from checkpoints
            for i in range(self.n_estimators):
                tmp_loader_cfg = self.loader_cfg
                tmp_loader_cfg["sampler"] = BootstrapSampler(len(data), i, self.bootstrap, self.frac_examples)

                self.estimators_.append(
                    Model(loader= tmp_loader_cfg, training_file="training_{}.jsonl".format(i), *self.args, **self.kwargs)
                )
                self.estimators_[i].fit(data)

        else:
            self.estimators_ = nn.ModuleList([
                Model(training_file="training_{}.jsonl".format(i), *self.args, **self.kwargs) for i in range(self.n_estimators)
            ])

            if self.train_method == "wagging":
                w_tensor = -torch.log( torch.randint(low=1,high=1000, size=(len(data), self.n_estimators)) / 1000.0 )
                w_dataset = WeightedDataset(data, w_tensor)
                super().fit(w_dataset)
            else:
                w_tensor = torch.poisson(torch.ones(size=(len(data), self.n_estimators)))
                w_dataset = WeightedDataset(data, w_tensor)
                super().fit(w_dataset)
