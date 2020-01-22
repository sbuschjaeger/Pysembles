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
from Utils import apply_in_batches, cov, weighted_mse_loss, weighted_squared_hinge_loss

class NNEnsemble(SKEnsemble):
    def __init__(self, n_estimators = 5, regularizer = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.regularizer = regularizer

    def fit(self, X, y):
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
            if self.x_test is not None:
                o_str = "epoch,loss,train-accuracy,reg,avg-train-accuracy,test-accuracy,avg-test-accuracy"
            else:
                o_str = "epoch,loss,train-accuracy,reg,avg-train-accuracy"
            outfile.write(o_str + "\n")
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_reg = 0
            n_correct = 0
            avg_n_correct = 0
            example_cnt = 0
            batch_cnt = 0

            with tqdm(total=len(train_loader.dataset), ncols=145, disable= not self.verbose) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)
                    
                    optimizer.zero_grad()
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

                    if self.regularizer is not None and not "reg_type" in self.regularizer:
                        base_preds = torch.stack(base_preds, dim=1)
                        B = base_preds.shape[0]
                        T = base_preds.shape[1]
                        K = base_preds.shape[2]
                        base_preds = torch.reshape(base_preds, (B, T*K))
                        c = cov(base_preds, bias=True, rowvar=False)

                        if self.loss_function == weighted_mse_loss or self.loss_function == nn.MSELoss:
                            idx_mat = torch.eye(K).cuda()
                            idx_mat = idx_mat.repeat(B,T,T)
                            D = 1.0/(self.n_classes_*T**2)*idx_mat
                        elif self.loss_function == weighted_squared_hinge_loss:
                            #f_bar = prediction.type(torch.cuda.FloatTensor)
                            target_one_hot = 2*torch.nn.functional.one_hot(target, num_classes = self.n_classes_).type(torch.cuda.FloatTensor) - 1.0
                            active = target_one_hot*f_bar
                            c1 = active < 0
                            c2 = active > 1
                            c3 = (0 < active) * (active < 1)
                            active[c1] = 0
                            active[c2] = 0
                            active[c3] = 1
                            # print(active.shape)
                            active = active.repeat(1,T).repeat(T*K,1,1).view(B,T*K,T*K)
                            # print(active.shape)
                            idx_mat = torch.eye(K).cuda()
                            idx_mat = idx_mat.repeat(B,T,T)
                            # print(idx_mat.shape)
                            # asdf
                            D = 1.0/(self.n_classes_*T**2)*idx_mat*active
                        else: #if self.regularizer["type"] == "exact":
                            def L_beta(h):
                                # TODO ENFORCE REDUCTION = NONE  IN LOSS
                                f_beta = h.view((B, T, K)).mean(dim=1)
                                tmp_loss = self.loss_function(f_beta, target)
                                return tmp_loss.mean()
                            
                            tmp_loss = L_beta(base_preds)
                            first_deriv = grad(tmp_loss, base_preds, create_graph=True)[0]
                            first_deriv = K*B*first_deriv.sum(dim=0) # TODO CHECK THIS
                            D = []
                            for di in first_deriv:
                                hessian = grad(di, base_preds, create_graph=False, retain_graph=True)[0]
                                D.append(hessian)
                            D = torch.stack(D, dim=1).mean(dim=0)
                            regularizer = 1.0/2.0*(D*c).sum()
                        # else:
                        #     D = torch.ones_like(c)
                        regularizer = 1.0/2.0*(D.detach()*c).sum()
                        l_reg = self.regularizer["lambda"]
                    elif self.regularizer is not None and "reg_type" in self.regularizer:
                        regularizer = 100. * avg_n_correct/(self.n_estimators*example_cnt)
                        l_reg = self.regularizer["lambda"]
                    else:
                        regularizer = torch.tensor(0)
                        l_reg = 0

                    total_reg += regularizer
                    loss = loss.mean() + l_reg * regularizer

                    loss.backward()
                    optimizer.step()

                    desc = "[{}/{}] loss {:4.3f} acc {:4.2f} avg acc {:4.2f} reg {:10.8f}".format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        100. * avg_n_correct/(self.n_estimators*example_cnt),
                        total_reg/batch_cnt
                    )

                    pbar.set_description(desc)
                
                if self.x_test is not None:
                    output = apply_in_batches(self, self.x_test)
                    accuracy_test_apply = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    output_proba = self.predict_proba(self.x_test)
                    accuracy_test_proba = accuracy_score(np.argmax(output_proba, axis=1),self.y_test)*100.0

                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    desc = "[{}/{}] loss {:4.3f} acc {:4.2f} avg acc {:4.2f} reg {:10.8f} test acc {:4.3f} avg test acc {:4.3f}, test acc proba {:4.3f}".format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        100. * avg_n_correct/(self.n_estimators*example_cnt),
                        total_reg/batch_cnt,
                        accuracy_test_apply,
                        np.mean(all_accuracy_test), 
                        accuracy_test_proba
                    )

                    pbar.set_description(desc)
            
            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                if self.x_test is not None:
                    output= self.predict_proba(self.x_test)
                    # output = apply_in_batches(self, self.x_test)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0
                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    o_str = "{},{},{},{},{},{},{}\n".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0 * n_correct/example_cnt, 
                        total_reg/batch_cnt,
                        100. * avg_n_correct/(self.n_estimators*example_cnt),
                        accuracy_test,
                        np.mean(all_accuracy_test)
                    )
                else:
                    o_str = "{},{},{},{},{}\n".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0 * n_correct/example_cnt, 
                        total_reg/batch_cnt,
                        100. * avg_n_correct/(self.n_estimators*example_cnt)
                    )
                outfile.write(o_str)
                if epoch % 10 == 0:
                    torch.save(self.state_dict(), os.path.join(self.out_path, 'model_{}.checkpoint'.format(epoch)))
                