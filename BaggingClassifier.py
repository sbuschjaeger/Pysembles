#!/usr/bin/env python3

import numpy as np
import torch

from sklearn.utils.multiclass import unique_labels

from Models import SKLearnModel
from Models import StagedEnsemble

class BaggingClassifier(StagedEnsemble):
    def __init__(self, optimizer_dict, scheduler_dict, loss_function, generate_model, 
                 verbose = True, out_path = None,  n_estimators = 5, frac_samples = 1, 
                 bootstrap = True, x_test = None, y_test = None):
        super().__init__()
        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict
        self.loss_function = loss_function
        self.generate_model = generate_model
        self.verbose = verbose
        self.out_path = out_path
        self.n_estimators = n_estimators
        self.frac_samples = frac_samples
        self.bootstrap = bootstrap

    def fit(self, X, y): 
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        self.estimators_ = [
            SKLearnModel(
                optimizer_dict = self.optimizer_dict, 
                scheduler_dict = self.scheduler_dict, 
                loss_function = self.loss_function, 
                generate_model = self.generate_model, 
                verbose = self.verbose, 
                out_path = self.out_path) for i in range(self.n_estimators)
        ]

        for idx, est in enumerate(self.estimators_):
            np.random.seed(idx)

            idx_array = [i for i in range(len(y))]
            idx_sampled = np.random.choice(
                idx_array, 
                size=int(self.frac_samples*len(idx_array)), 
                replace=self.bootstrap
            )

            X_sampled = X[idx_sampled,] 
            y_sampled = y[idx_sampled]
            est.fit(X_sampled, y_sampled)
