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

from .Models import SKLearnModel
# from Models import Flatten
from .BinarisedNeuralNetworks import BinaryTanh
from .BinarisedNeuralNetworks import BinaryLinear


# TODO THIS CLASSIFIER NEEDS TO BE UPDATED
class StackingClassifier(SKLearnModel):
    def __init__(self, optimizer_dict, scheduler_dict, loss_function, base_estimator,verbose = True, out_path = None,  n_estimators = 5, l_reg = 0):
        super().__init__(optimizer_dict, scheduler_dict, loss_function, base_estimator, verbose, out_path)
        self.n_estimators = n_estimators
        self.l_reg = l_reg

    def forward(self, x, est_idx = None):
        self.base_preds = [est(x) for est in self.estimators_]
        # print("base_preds: ", base_preds.shape)
        return self.classifier(torch.stack(self.base_preds, dim=1))

        #w_normalized = nn.functional.softmax(self.estimators_weights_, dim=0).cuda()
        # w_normalized = self.estimators_weights_

        # if est_idx is None: 
        #     self.base_preds = torch.stack([est(x) for est in self.estimators_])
        #     weighted_preds = self.base_preds*w_normalized[:,None,None]
            
        #     tmp = torch.sum(weighted_preds,dim=0) 
        #     return tmp
        # else:
        #     return self.estimators_[est_idx](x)*w_normalized[est_idx]

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = nn.ModuleList([ self.generate_model() for _ in range(self.n_estimators)])
        self.X_ = X
        self.y_ = y

        if self.model_type == "binary":
            self.classifier = nn.Sequential(
                #Flatten(),
                nn.BatchNorm1d(self.n_classes_*self.n_estimators),
                BinaryTanh(),
                BinaryLinear(self.n_classes_*self.n_estimators,self.n_classes_)
            )
        else:
            self.classifier = nn.Sequential(
                #Flatten(),
                nn.Linear(self.n_classes_*self.n_estimators,self.n_classes_)
            )

        x_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)  
        y_tensor = y_tensor.type(torch.LongTensor) 

        data = torch.utils.data.TensorDataset(x_tensor,y_tensor)
        optimizer = self.optimizer_dict["method"](self.parameters(), **self.optimizer_dict["config"])
        
        if self.scheduler_dict is not None:
            scheduler = self.scheduler_dict["method"](optimizer, **self.scheduler_dict["config"])
        else:
            scheduler = None

        cuda_cfg = {'num_workers': 1, 'pin_memory': True} 
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.optimizer_dict["batch_size"], 
            shuffle=True, 
            **cuda_cfg
        )

        self.cuda()
        
        if self.out_path is not None:
            outfile = open(self.out_path, "w", 1)
            o_str = "epoch,loss,train-accuracy," + self.log_header()
            outfile.write(o_str + "\n")

        for epoch in range(self.optimizer_dict["epochs"]):
            epoch_loss = 0
            n_correct = 0
            example_cnt = 0
            with tqdm(total=len(train_loader.dataset), ncols=110, disable= not self.verbose) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)

                    optimizer.zero_grad()
                    output = self(data)
                    accuracy = (output.argmax(1) == target).type(torch.cuda.FloatTensor)
                    loss = self.loss_function(output, target)
                    epoch_loss += loss.sum().item()
                    loss = loss.mean() 
                    n_correct += accuracy.sum().item()

                    for predi in self.base_preds:
                        loss += self.l_reg*self.loss_function(predi, target).mean()

                    loss.backward()
                    optimizer.step()

                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]

                    pbar.set_description('epoch [{}/{}] - loss {:.6f} - acc {:.6f}'.format(epoch, self.optimizer_dict["epochs"]-1, epoch_loss/example_cnt, 100. * n_correct/example_cnt))
            
            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                outfile.write("{},{:4f},{},{}\n".format(epoch, epoch_loss/example_cnt, 100.0*n_correct/example_cnt, self.log(x_tensor,y_tensor)))