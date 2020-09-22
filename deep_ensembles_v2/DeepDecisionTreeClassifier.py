#!/usr/bin/env python3

import os
import numpy as np
import torch
import random
import copy

from torch import nn
from torch.autograd import Variable

from tqdm import tqdm

from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

from .Utils import apply_in_batches, TransformTensorDataset
from .Models import SKLearnModel

class DeepDecisionTreeClassifier(SKLearnModel):
    def __init__(self, split_estimator, leaf_estimator, depth, soft=False, *args, **kwargs):
        super().__init__(base_estimator = lambda: None, *args, **kwargs)
        
        self.depth = depth
        self.split_estimator = split_estimator
        self.leaf_estimator = leaf_estimator
        self.soft = soft

        self.layers_ = []
        self.n_inner = int((2**(self.depth+1) - 1)/2)
        self.n_leafs = 2**self.depth
        for i in range(self.n_inner):
             self.layers_.append(self.split_estimator())
        
        for i in range(self.n_leafs):
             self.layers_.append(self.leaf_estimator())
        self.layers_ = nn.Sequential(*self.layers_)
        
        cur_path = [[0]]
        for i in range(self.depth):
            tmp_path = []
            for p in cur_path:
                p1 = p.copy()
                p2 = p.copy()
                p1.append( 2*p[-1] + 1 )
                p2.append( 2*p[-1] + 2 )
                tmp_path.append(p1)
                tmp_path.append(p2)
            cur_path = tmp_path
        self.all_pathes = cur_path

    def fit(self, X, y, sample_weight = None): 
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        #self.layers_ = nn.Sequential(*[self.split_estimator(), self.leaf_estimator(), self.leaf_estimator()])

        x_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)  
        y_tensor = y_tensor.type(torch.LongTensor) 

        if sample_weight is not None:
            sample_weight = len(y)*sample_weight/np.sum(sample_weight)
            w_tensor = torch.tensor(sample_weight)
            w_tensor = w_tensor.type(torch.FloatTensor)
            data = TransformTensorDataset(x_tensor,y_tensor,w_tensor,transform=self.transformer)
        else:
            w_tensor = None
            data = TransformTensorDataset(x_tensor,y_tensor,transform=self.transformer)

        self.X_ = X
        self.y_ = y

        optimizer = self.optimizer_method(self.parameters(), **self.optimizer)
        
        if self.scheduler_method is not None:
            scheduler = self.scheduler_method(optimizer, **self.scheduler)
        else:
            scheduler = None

        cuda_cfg = {'num_workers': 1, 'pin_memory': True} 
        
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size, 
            shuffle=True, 
            **cuda_cfg
        )

        self.cuda()
        self.train()
        if self.out_path is not None:
            #file_cnt = sum([1 if "training" in fname else 0 for fname in os.listdir(self.out_path)])
            outfile = open(self.out_path + "/" + self.training_csv, "w", 1)
            if self.x_test is not None:
                o_str = "epoch,loss,train-accuracy,test-accuracy"
            else:
                o_str = "epoch,loss,train-accuracy"

            outfile.write(o_str + "\n")

        for epoch in range(self.epochs):
            epoch_loss = 0
            n_correct = 0
            example_cnt = 0
            batch_cnt = 0
            self.cnts = [0 for i in range(len(self.layers_))]

            with tqdm(total=len(train_loader.dataset), ncols=135, disable = not self.verbose) as pbar:
                for batch in train_loader:
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)
                    if sample_weight is not None:
                        weights = batch[2]
                        weights = weights.cuda()
                        weights = Variable(weights)

                    optimizer.zero_grad()
                    output = self(data)
                    unweighted_acc = (output.argmax(1) == target).type(torch.cuda.FloatTensor)
                    if sample_weight is not None: 
                        loss = self.loss_function(output, target, weights)
                        epoch_loss += loss.sum().item()
                        loss = loss.mean()

                        weighted_acc = unweighted_acc*weights
                        n_correct += weighted_acc.sum().item()
                    else:
                        loss = self.loss_function(output, target)
                        epoch_loss += loss.sum().item()
                        loss = loss.mean()
                        n_correct += unweighted_acc.sum().item()
                    
                    loss.backward()
                    optimizer.step()

                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]
                    batch_cnt += 1
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt
                    )
                    pbar.set_description(desc)
                
                print("")
                print("Leaf cnts are {} with total sum of {}".format(self.cnts[self.n_inner:], sum(self.cnts[self.n_inner:])))
                #print("Leaf cnts are {} with total sum of {}".format(self.cnts, sum(self.cnts)))
                if self.x_test is not None:
                    # output = apply_in_batches(self, self.x_test)
                    # accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    # output = apply_in_batches(self, self.x_test, batch_size = self.batch_size)
                    # accuracy_test_apply = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    output_proba = self.predict_proba(self.x_test)
                    accuracy_test_proba = accuracy_score(np.argmax(output_proba, axis=1),self.y_test)*100.0

                    desc = '[{}/{}] loss {:2.4f} train acc {:2.4f} test acc {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        # accuracy_test_apply,
                        accuracy_test_proba
                    )
                    pbar.set_description(desc)

            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                avg_loss = epoch_loss/example_cnt
                accuracy = 100.0*n_correct/example_cnt

                if self.x_test is not None:
                    output = apply_in_batches(self, self.x_test, batch_size = self.batch_size)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0
                    outfile.write("{},{},{},{}\n".format(epoch, avg_loss, accuracy, accuracy_test))
                else:
                    outfile.write("{},{},{}\n".format(epoch, avg_loss, accuracy))

    def forward(self, x):
        # Execute all models; This can be improved
        all_preds = [l(x) for l in self.layers_]
        path_preds = []
        for path in self.all_pathes:
            # print(path)
            pred = torch.tensor(1.0)
            for i in range(len(path[:-1])):
                cur_node = path[i]
                next_node = path[i+1]
                n_pred = all_preds[cur_node]

                if not self.soft:
                    tmp = n_pred.clone()
                    tmp[tmp >= 0.5] = 1.0
                    tmp[tmp < 0.5] = 0.0
                    n_pred = tmp

                if cur_node == 0:
                    self.cnts[cur_node] += x.shape[0]
                else:
                    self.cnts[cur_node] += (pred != 0).sum().item()
                
                if (next_node % 2) == 0:
                    pred = n_pred * pred
                else:
                    pred = (1.0 - n_pred) * pred
            
            self.cnts[path[-1]] += (pred != 0).sum().item()
            pred = pred * all_preds[path[-1]]
            path_preds.append(pred)
        # asdf
        tmp = torch.stack(path_preds)
        
        return tmp.sum(dim = 0)
