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
from Utils import apply_in_batches, cov, weighted_mse_loss, weighted_squared_hinge_loss, is_same_func

class NaiveEnsemble(SKEnsemble):
    def __init__(self, n_estimators = 5, regularizer = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.regularizer = regularizer

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

        optimizers = [ self.optimizer_method(self.parameters(), **self.optimizer)  for i in range(self.n_estimators) ]
        if self.scheduler_method is not None:
            schedulers = [ self.scheduler_method(optimizers[i], **self.scheduler) for i in range(self.n_estimators) ]
        else:
            schedulers = None

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

        if self.regularizer is not None and "lambda" in self.regularizer:
            l_reg = self.regularizer["lambda"]
        else:
            l_reg = 0

        for epoch in range(self.epochs):
            n_correct = 0
            avg_n_correct = 0
            example_cnt = 0
            batch_cnt = 0

            total_loss = 0
            ensemble_losses = [0 for _ in range(self.n_estimators)]
            ensemble_reg = [0 for _ in range(self.n_estimators)]
            ensemble_reg_loss = [0 for _ in range(self.n_estimators)]
            diversity = 0

            with tqdm(total=len(train_loader.dataset), ncols=145, disable= not self.verbose) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)

                    for opt in optimizers:
                        opt.zero_grad()

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

                    for i, pred in enumerate(base_preds):
                        if (is_same_func(self.loss_function, weighted_mse_loss) or
                                is_same_func(self.loss_function, nn.MSELoss)):
                            D = 2.0
                        elif is_same_func(self.loss_function, weighted_squared_hinge_loss):
                            target_one_hot = 2*torch.nn.functional.one_hot(target, num_classes = self.n_classes_).type(torch.cuda.FloatTensor) - 1.0
                            active = target_one_hot*f_bar
                            c1 = active < 0
                            c2 = active > 1
                            c3 = (0 < active) * (active < 1)
                            active[c1] = 0
                            active[c2] = 0
                            active[c3] = 1
                            # print(active.shape)
                            # active = active.repeat(1,T).repeat(T*K,1,1).view(B,T*K,T*K)
                            # idx_mat = torch.eye(K).cuda()
                            # idx_mat = idx_mat.repeat(B,T,T)
                            D = 2.0*active
                            # print(D)
                        else: #if self.regularizer["type"] == "exact":
                            pass
                            # TODO
                            # def L_beta(h):
                            #     # TODO ENFORCE REDUCTION = NONE  IN LOSS
                            #     f_beta = h.view((B, T, K)).mean(dim=1)
                            #     tmp_loss = self.loss_function(f_beta, target)
                            #     return tmp_loss.mean()
                            
                            # tmp_loss = L_beta(stacked_base_preds)
                            # first_deriv = grad(tmp_loss, stacked_base_preds, create_graph=True)[0]
                            # first_deriv = K*B*first_deriv.sum(dim=0) # TODO CHECK THIS
                            # D = []
                            # for di in first_deriv:
                            #     hessian = grad(di, stacked_base_preds, create_graph=False, retain_graph=True)[0]
                            #     D.append(hessian)
                            # D = torch.stack(D, dim=1).mean(dim=0)
                            #regularizer = 1.0/2.0*(D*c).sum()

                        diff = pred - f_bar
                        reg = 1.0/(2*self.n_classes_*self.n_estimators) * ((D * diff) * diff).sum(dim=1)
                        # print(diff.shape)
                        # print(D.shape)
                        # reg = diff.t() * D * diff
                        # print("REG:", reg)
                        # print("REG SHAPE:", reg.shape)
                        # dim=1: over classes
                        # dim=0: over batches
                        div = reg.mean()
                        diversity += div.detach()
                        
                        if self.regularizer is not None:
                            l_reg = self.regularizer["lambda"]
                            regularizer = div
                        else:
                            l_reg = 0
                            regularizer = torch.tensor(0)

                        i_loss = self.loss_function(pred, target)

                        ensemble_losses[i] += i_loss.sum().item()
                        # print("REG:", regularizer)
                        # print(regularizer.shape)
                        ensemble_reg_loss[i] += i_loss.mean().item() + l_reg * regularizer.item()
                        ensemble_reg[i] += regularizer.item()

                        reg_loss = loss.mean() + l_reg * regularizer

                        if i == self.n_estimators - 1:
                            reg_loss.backward(retain_graph=False)
                        else:
                            reg_loss.backward(retain_graph=True)
                        optimizers[i].step()
                    # stacked_base_preds = torch.stack(base_preds, dim=1)
                    # B = stacked_base_preds.shape[0]
                    # T = stacked_base_preds.shape[1]
                    # K = stacked_base_preds.shape[2]
                    # stacked_base_preds = torch.reshape(stacked_base_preds, (B, T*K))
                    # c = cov(stacked_base_preds, bias=True, rowvar=False)

                    # if (is_same_func(self.loss_function, weighted_mse_loss) or
                    #     is_same_func(self.loss_function, nn.MSELoss)):
                    #     idx_mat = torch.eye(K).cuda()
                    #     idx_mat = idx_mat.repeat(B,T,T)
                    #     D = 1.0/(self.n_classes_*T*(T-1))*idx_mat
                    # elif is_same_func(self.loss_function, weighted_squared_hinge_loss):
                    #     target_one_hot = 2*torch.nn.functional.one_hot(target, num_classes = self.n_classes_).type(torch.cuda.FloatTensor) - 1.0
                    #     active = target_one_hot*f_bar
                    #     c1 = active < 0
                    #     c2 = active > 1
                    #     c3 = (0 < active) * (active < 1)
                    #     active[c1] = 0
                    #     active[c2] = 0
                    #     active[c3] = 1
                    #     active = active.repeat(1,T).repeat(T*K,1,1).view(B,T*K,T*K)
                    #     idx_mat = torch.eye(K).cuda()
                    #     idx_mat = idx_mat.repeat(B,T,T)
                    #     D = 1.0/(self.n_classes_*T*(T-1))*idx_mat*active
                    # else: #if self.regularizer["type"] == "exact":
                    #     def L_beta(h):
                    #         # TODO ENFORCE REDUCTION = NONE  IN LOSS
                    #         f_beta = h.view((B, T, K)).mean(dim=1)
                    #         tmp_loss = self.loss_function(f_beta, target)
                    #         return tmp_loss.mean()
                        
                    #     tmp_loss = L_beta(stacked_base_preds)
                    #     first_deriv = grad(tmp_loss, stacked_base_preds, create_graph=True)[0]
                    #     first_deriv = K*B*first_deriv.sum(dim=0) # TODO CHECK THIS
                    #     D = []
                    #     for di in first_deriv:
                    #         hessian = grad(di, stacked_base_preds, create_graph=False, retain_graph=True)[0]
                    #         D.append(hessian)
                    #     D = torch.stack(D, dim=1).mean(dim=0)
                    #     #regularizer = 1.0/2.0*(D*c).sum()
                    
                    # diversity += 1.0/2.0*(D.detach()*c.detach()).sum()
                    # if self.regularizer is not None:
                    #     print(D.detach()*c)
                    #     print(c.shape)
                    #     asdf
                    #     l_reg = self.regularizer["lambda"]
                    #     regularizer = - 1.0/2.0*(D.detach()*c).sum()
                    # else:
                    #     regularizer = torch.tensor(0)
                    #     l_reg = 0

                    # for i, pred in enumerate(base_preds):
                    #     loss = self.loss_function(pred, target)

                    #     ensemble_losses[i] += loss.sum().item()
                    #     ensemble_reg[i] += regularizer.item()

                    #     loss = loss.mean() + l_reg * regularizer
                    #     if i == self.n_estimators - 1:
                    #         loss.backward(retain_graph=False)
                    #     else:
                    #         loss.backward(retain_graph=True)
                    #     optimizers[i].step()

                    desc = '[{}/{}] e-loss {:2.4f} e-train-acc {:2.3f} i-reg-loss {:2.4f} avg-i-acc {:2.3f} div {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt, 
                        np.mean(ensemble_reg_loss) / batch_cnt, 
                        100 * avg_n_correct / (example_cnt *self.n_estimators),
                        diversity / (batch_cnt * self.n_estimators)
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

                    desc = '[{}/{}] loss {:2.4f} acc {:2.3f}  avg-acc {:2.3f} div {:2.4f} test acc {:4.3f} avg test acc {:4.3f}'.format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt, 
                        np.mean(ensemble_reg_loss) / batch_cnt, 
                        diversity / (batch_cnt * self.n_estimators),
                        accuracy_test,
                        np.mean(all_accuracy_test)
                    )

                    pbar.set_description(desc)

            for s in schedulers:
                s.step()
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