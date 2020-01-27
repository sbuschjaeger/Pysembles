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

from Models import SKEnsemble
from Utils import apply_in_batches, cov, weighted_mse_loss, weighted_squared_hinge_loss, is_same_func, weighted_cross_entropy

class GNCLClassifier(SKEnsemble):
    def __init__(self, n_estimators = 5, l_reg = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.l_reg = l_reg

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])

        self.X_ = X
        self.y_ = y

        x_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)  
        y_tensor = y_tensor.type(torch.LongTensor) 

        data = torch.utils.data.TensorDataset(x_tensor,y_tensor)

        optimizer = self.optimizer_method(self.parameters(), **self.optimizer)
        scheduler = self.scheduler_method(optimizer, **self.scheduler)

        # optimizers = [ self.optimizer_method(self.estimators_[i].parameters(), **self.optimizer)  for i in range(self.n_estimators) ]
        # if self.scheduler_method is not None:
        #     schedulers = [ self.scheduler_method(optimizers[i], **self.scheduler) for i in range(self.n_estimators) ]
        # else:
        #     schedulers = None

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
            if self.x_test is not None:
                header = "epoch,loss,train-accuracy,avg-train-accuracy,test-accuracy,avg-test-accuracy"
            else:
                header = "epoch,loss,train-accuracy,avg-train-accuracy"
            
            for i in range(self.n_estimators):
                header += ",loss_" + str(i)
            for i in range(self.n_estimators):
                header += ",reg_" + str(i)
            outfile.write(header + "\n")

        for epoch in range(self.epochs):
            n_correct = 0
            avg_n_correct = 0
            example_cnt = 0
            batch_cnt = 0

            total_loss = 0
            ensemble_losses = [0 for _ in range(self.n_estimators)]
            ensemble_reg = [0 for _ in range(self.n_estimators)]
            diversity = 0

            with tqdm(total=len(train_loader.dataset), ncols=145, disable= not self.verbose) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)

                    optimizer.zero_grad()
                    # for opt in optimizers:
                    #     opt.zero_grad()

                    f_bar, base_preds = self.forward_with_base(data)
                    
                    for pred in base_preds:
                        acc = (pred.argmax(1) == target).type(torch.cuda.FloatTensor)
                        avg_n_correct += acc.sum().item()

                    accuracy = (f_bar.argmax(1) == target).type(torch.cuda.FloatTensor)
                    loss = self.loss_function(f_bar, target)
                    total_loss += loss.sum().item()
                    n_correct += accuracy.sum().item()
                    
                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]
                    batch_cnt += 1

                    if (is_same_func(self.loss_function, weighted_mse_loss) or is_same_func(self.loss_function, nn.MSELoss)):
                        n_classes = pred.shape[1]
                        n_preds = pred.shape[0]

                        eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
                        D = 2.0*eye_matrix
                    elif (is_same_func(self.loss_function, weighted_cross_entropy) or is_same_func(self.loss_function, nn.CrossEntropyLoss)):
                        n_preds = pred.shape[0]
                        n_classes = pred.shape[1]
                        
                        f_bar_softmax = nn.functional.softmax(f_bar,dim=1)
                        D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
                        diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
                        D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
                        # for i in range(n_classes):
                        #     D[:,i,i] = diag_vector[:,i]
                    else:
                        # TODO Use autodiff do compute second derivative for given loss function
                        D = 1.0

                    sum_losses = None
                    for i, pred in enumerate(base_preds):
                        # TODO MAYBE NOT DETACH THIS
                        diff = pred - f_bar #.detach()
                        covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
                        reg = -0.5 * covar.mean()
                        diversity += reg.item()

                        i_loss = self.loss_function(pred, target)
                        ensemble_losses[i] += i_loss.sum().item()
                        ensemble_reg[i] += reg.item()

                        i_mean = i_loss.mean()
                        #reg_loss = -2 * i_mean * self.l_reg * reg 
                        reg_loss = i_mean + self.l_reg * reg 
                        
                        # if reg_loss < 0:
                        #     reg_loss = i_mean

                        if sum_losses is not None:
                            sum_losses += reg_loss
                        else:
                            sum_losses = reg_loss
                        # if i == self.n_estimators - 1:
                        #     reg_loss.backward(retain_graph=False)
                        # else:
                        #     reg_loss.backward(retain_graph=True)
                        #optimizers[i].step()
                    sum_losses.backward()
                    optimizer.step()

                    desc = '[{}/{}] loss {:2.4f} acc {:2.3f} bias {:2.4f} var {:2.4f} avg train acc {:4.3f}'.format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt, 
                        np.mean(ensemble_losses) / example_cnt, 
                        diversity / (batch_cnt * self.n_estimators), 
                        100. * avg_n_correct/(example_cnt * self.n_estimators)
                    )
                    pbar.set_description(desc)
            
                if self.x_test is not None:
                    output = apply_in_batches(self, self.x_test)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    desc = '[{}/{}] loss {:2.4f} acc {:2.3f} bias {:2.3f} var {:2.4f} avg train acc {:4.3f} test acc {:4.3f}'.format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt, 
                        np.mean(ensemble_losses) / example_cnt, 
                        diversity / (batch_cnt * self.n_estimators),
                        100. * avg_n_correct/(example_cnt * self.n_estimators),
                        accuracy_test
                    )

                    pbar.set_description(desc)

            scheduler.step()
            # for s in schedulers:
            #     s.step()
            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                if self.x_test is not None:
                    output = apply_in_batches(self, self.x_test)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    out_str = "{},{},{},{}".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0*n_correct/example_cnt,
                        100. * avg_n_correct/(self.n_estimators*example_cnt),
                        accuracy_test,
                        np.mean(all_accuracy_test)
                    )
                else:
                    out_str = "{},{},{},{}".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0*n_correct/example_cnt,
                        100. * avg_n_correct/(self.n_estimators*example_cnt)
                    )
                
                for l in ensemble_losses:
                    out_str += "," + str(l/example_cnt)
                for r in ensemble_reg:
                    out_str += "," + str(r/example_cnt)

                outfile.write(out_str + "\n")