#!/usr/bin/env python3
import os

import numpy as np
from tqdm import tqdm

from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad

from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from Models import SKLearnBaseModel
from Utils import apply_in_batches, cov, Flatten, weighted_cross_entropy, weighted_mse_loss

class DiverseNN(SKLearnBaseModel):
    def __init__(self, regularizer = None, *args, **kwargs):
        # print(kwargs.keys())
        super().__init__(*args, **kwargs)
        self.regularizer = regularizer

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        self.layers_ = self.base_estimator()

        self.X_ = X
        self.y_ = y

        x_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)  
        y_tensor = y_tensor.type(torch.LongTensor) 

        data = torch.utils.data.TensorDataset(x_tensor,y_tensor)
        optimizer = self.optimizer_method(self.parameters(), **self.optimizer)
        
        if self.scheduler_method is not None:
            scheduler = self.scheduler_method(optimizer, **self.scheduler)
        else:
            scheduler = None

        cuda_cfg = {'num_workers': 0, 'pin_memory': True} 
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size, 
            shuffle=True, 
            **cuda_cfg
        )

        self.cuda()
        
        if self.out_path is not None:
            outfile = open(self.out_path + "/training.csv", "w", 1)
            o_str = "epoch,loss,train-accuracy,reg"
            outfile.write(o_str + "\n")
        
        for epoch in range(self.epochs):
            total_loss = 0
            total_reg = 0
            n_correct = 0
            example_cnt = 0
            batch_cnt = 0

            with tqdm(total=len(train_loader.dataset), ncols=145, disable= not self.verbose) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)

                    optimizer.zero_grad()
                    output = self(data)

                    accuracy = (output.argmax(1) == target).type(torch.cuda.FloatTensor)
                    loss = self.loss_function(output, target)
                    total_loss += loss.sum().item()
                    n_correct += accuracy.sum().item()
                    
                    if self.regularizer is not None:
                        base_preds = self.pre_output
                        
                        y = target.view(-1,1,1)
                        yhat = output 
                        C = self.n_classes_
                        B = base_preds.shape[0]
                        T = self.no_filters_before_flatten
                        K = int(base_preds.shape[1] / T)
                        c = cov(base_preds, bias=True, rowvar=False)

                        #if self.regularizer["type"] == "exact":
                        if self.loss_function == weighted_cross_entropy or self.loss_function == nn.CrossEntropyLoss:
                            W = self.layers_[-1].weight.t().repeat(B,1,1)
                            Wy = W * yhat.unsqueeze(1)
                            B = torch.bmm(W, W.view(-1, W.shape[2], W.shape[1]))
                            
                            inner = torch.bmm(W, yhat.unsqueeze(2)).squeeze(2)
                            outer = torch.bmm(inner.unsqueeze(2), inner.unsqueeze(1))
                            D = (outer - B).mean(dim=0).detach()
                        elif self.loss_function == weighted_mse_loss or self.loss_function == nn.MSELoss:
                            W = self.layers_[-1].weight.t().repeat(B,1,1)
                            WW = torch.bmm(W, W.view(-1, W.shape[2], W.shape[1]))
                            D = 2*1.0/self.n_classes_*WW.mean(dim=0) #.detach()
                        else:
                            #if self.regularizer["type"] == "exact":
                            def L_beta(h):
                                # TODO ENFORCE OUTPUT IS LINEAR
                                f_beta  = F.linear(base_preds, self.layers_[-1].weight, self.layers_[-1].bias)
                                # TODO ENFORCE REDUCTION = NONE  IN LOSS
                                #f_beta = h.view((B, T, K)).mean(dim=1)
                                tmp_loss = self.loss_function(f_beta, target)
                                return tmp_loss.mean()

                            tmp_loss = L_beta(base_preds)
                            first_deriv = grad(tmp_loss, base_preds, create_graph=True)[0]
                            first_deriv = K*B*first_deriv.sum(dim=0) # TODO CHECK THIS

                            D = []
                            for di in first_deriv:
                                hessian, = grad(di, base_preds, create_graph=False, retain_graph=True)
                                D.append(hessian)
                            D = torch.stack(D, dim=1).mean(dim=0).detach()

                        regularizer = 1.0/2.0*(D*c).sum()
                        #print(c.shape)
                        l_reg = self.regularizer["lambda"]
                    else:
                        regularizer = torch.tensor(0)
                        l_reg = 0

                    total_reg += regularizer
                    loss = loss.mean() + l_reg * regularizer

                    loss.backward()
                    optimizer.step()

                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]
                    batch_cnt += 1

                    desc = "[{}/{}] loss {:2.4f} acc {:2.3f} reg {:2.3f}".format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        total_reg/batch_cnt
                    )

                    pbar.set_description(desc)
                
                if self.x_test is not None:
                    output = apply_in_batches(self, self.x_test)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    desc = "[{}/{}] loss {:2.4f} acc {:2.3f} reg {:2.3f} tacc {:2.3f}".format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        total_reg/batch_cnt,
                        accuracy_test
                    )

                    pbar.set_description(desc)
            
            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                o_str = "{},{},{},{}\n".format(
                    epoch, 
                    total_loss/example_cnt, 
                    100.0*n_correct/example_cnt, 
                    total_reg/batch_cnt
                )
                outfile.write(o_str)
                if epoch % 10 == 0:
                    torch.save(self.state_dict(), os.path.join(self.out_path, 'model_{}.checkpoint'.format(epoch)))
    
    def forward(self, x):
        pre_output = None

        for l in self.layers_:
            if isinstance(l, Flatten):
                self.no_filters_before_flatten = x.shape[1]
                # print("BEFORE FLATTEN:", x.shape)
            pre_output = x
            x = l(x)

        self.pre_output = pre_output
        return x
