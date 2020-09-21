#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

from torch import nn
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Models import SKEnsemble 

class GradientBoostedNets(SKEnsemble):
    def __init__(self, optimizer_dict, scheduler_dict, loss_function, base_estimator,
                 verbose = True, out_path = None,  n_estimators = 5, l_reg = 0, reg_type = "none",
                 x_test = None, y_test = None):
        super().__init__()
        self.optimizer_dict = optimizer_dict
        self.scheduler_dict = scheduler_dict
        self.loss_function = loss_function
        self.base_estimator = base_estimator
        self.verbose = verbose
        self.out_path = out_path
        self.n_estimators = n_estimators
        self.l_reg = l_reg
        self.reg_type = reg_type
        self.x_test = x_test
        self.y_test = y_test
        
    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        self.estimators_ = [self.base_estimator() for _ in range(self.n_estimators)]

        self.X_ = X
        self.y_ = y

        x_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        y_tensor = y_tensor.type(torch.LongTensor) 

        data = torch.utils.data.TensorDataset(x_tensor,y_tensor)

        optimizers = [
            self.optimizer_dict["method"](self.estimators_[i].parameters(), **self.optimizer_dict["config"]) 
            for i in range(self.n_estimators)
        ]
        
        if self.scheduler_dict is not None:
            schedulers = [
                self.scheduler_dict["method"](optimizers[i],**self.scheduler_dict["config"]) 
                for i in range(self.n_estimators)
            ]
        else:
            schedulers = None

        cuda_cfg = {'num_workers': 1, 'pin_memory': True} 
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.optimizer_dict["batch_size"], 
            shuffle=True, 
            **cuda_cfg
        )

        for p in self.estimators_:
            p.cuda()
        self.cuda()
        
        if self.out_path is not None:
            outfile = open(self.out_path + "/training.csv", "w", 1)
            outfile.write("model,epoch,loss,reg,train-accuracy\n")

        for i in range(self.n_estimators):
            for epoch in range(self.optimizer_dict["epochs"]):
                total_loss = 0
                total_reg = 0
                n_correct = 0
                example_cnt = 0
                batch_cnt = 0
                
                with tqdm(total=len(train_loader.dataset), ncols=115, disable= not self.verbose) as pbar:
                    for batch_idx, batch in enumerate(train_loader):
                        data = batch[0]
                        target = batch[1]
                        data, target = data.cuda(), target.cuda()
                        data, target = Variable(data), Variable(target)
                        optimizers[i].zero_grad()
                        
                        if i > 0:
                            base_preds = [1.0/self.n_estimators*est(data) for est in self.estimators_[0:i]]
                            pred_combined = torch.sum(torch.stack(base_preds, dim=1),dim=1).detach() 
                            pred_combined += self.estimators_[i](data)
                        else:
                            pred_combined = self.estimators_[i](data)
                        
                        accuracy = (pred_combined.argmax(1) == target).type(torch.cuda.FloatTensor)
                        loss = self.loss_function(pred_combined, target)
                        total_loss += loss.sum().item()
                        n_correct += accuracy.sum().item()
                            
                        if self.l_reg > 0:
                            if self.reg_type == "cbound":
                                #loss += self.l_reg*self.cbound(data,target)
                                #regularizer = -loss.mean()/((loss.mean())**2)
                                regularizer = loss.var()/((loss.mean())**2)
                            elif self.reg_type == "var":
                                regularizer = torch.sqrt(loss.var())
                            else:
                                regularizer = torch.tensor(0)

                            total_reg += regularizer.item()
                            loss = self.l_reg * regularizer + loss.mean()
                        else:
                            loss = loss.mean()

                        loss.backward()
                        optimizers[i].step()
                        
                        pbar.update(data.shape[0])
                        example_cnt += data.shape[0]
                        batch_cnt += 1

                        desc = '[{}/{}] loss {:2.4f} acc {:2.3f} reg {:2.3f}'.format(
                            epoch, 
                            self.optimizer_dict["epochs"]-1, 
                            total_loss/example_cnt, 
                            100. * n_correct/example_cnt, 
                            total_reg/example_cnt
                        )
                        pbar.set_description(desc)
                
                if schedulers is not None:
                    schedulers[i].step()
                
                if self.out_path is not None:
                    out_str = "{},{},{},{},{}\n".format(
                        i,epoch, total_loss/example_cnt, total_reg/example_cnt, 100.0*n_correct/example_cnt
                    )
                    outfile.write(out_str)
