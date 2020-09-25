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

from .Models import SKEnsemble
from .Utils import apply_in_batches, cov, is_same_func, TransformTensorDataset
from .Losses import weighted_cross_entropy_with_softmax, weighted_mse_loss, weighted_cross_entropy

class GNCLClassifier(SKEnsemble):
    def __init__(self, n_estimators = 5, l_reg = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.l_reg = l_reg
        # self.l_mode = l_mode

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])

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
        scheduler = self.scheduler_method(optimizer, **self.scheduler)

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
            avg_n_correct = [0 for _ in range(self.n_estimators)]
            example_cnt = 0
            batch_cnt = 0

            total_loss = 0
            ensemble_losses = [0 for _ in range(self.n_estimators)]
            diversity = []
            ensemble_reg = [0 for _ in range(self.n_estimators)]

            with tqdm(total=len(train_loader.dataset), ncols=145, disable= not self.verbose) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)

                    if sample_weight is not None:
                        weights = batch[2]
                        weights = weights.cuda()
                        weights = Variable(weights)
                    
                    optimizer.zero_grad()
                    f_bar, base_preds = self.forward_with_base(data)

                    #diffs = []
                    for i, pred in enumerate(base_preds):
                        acc = (pred.argmax(1) == target).type(torch.cuda.FloatTensor)
                        avg_n_correct[i] += acc.sum().item()
                        # diff = pred - f_bar.detach()
                        # diffs.append(diff)

                    accuracy = (f_bar.argmax(1) == target).type(torch.cuda.FloatTensor)
                    loss = self.loss_function(f_bar, target)

                    if sample_weight is not None:
                        loss = loss * weights

                    total_loss += loss.sum().item()
                    n_correct += accuracy.sum().item()
                    
                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]
                    batch_cnt += 1
                    
                    # TODO I guess we should add weights here ?
                    if (is_same_func(self.loss_function, weighted_mse_loss) or is_same_func(self.loss_function, nn.MSELoss)):
                        n_classes = f_bar.shape[1]
                        n_preds = f_bar.shape[0]

                        eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
                        D = 2.0*eye_matrix
                    elif (is_same_func(self.loss_function, weighted_cross_entropy)):
                        n_classes = f_bar.shape[1]
                        n_preds = f_bar.shape[0]
                        D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
                        target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes).type(torch.cuda.FloatTensor)

                        eps = 1e-7
                        diag_vector = target_one_hot*(1.0/(f_bar**2+eps))
                        D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
                    elif (is_same_func(self.loss_function, weighted_cross_entropy_with_softmax) or is_same_func(self.loss_function, nn.CrossEntropyLoss)):
                        n_preds = f_bar.shape[0]
                        n_classes = f_bar.shape[1]
                        f_bar_softmax = nn.functional.softmax(f_bar,dim=1)
                        D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
                        diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
                        D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
                    # elif (is_same_func(self.loss_function, weighted_lukas_loss)):
                    #     # TODO DEBUG IF THIS IS ACTUALLY CORRECT
                    #     n_preds = pred.shape[0]
                    #     n_classes = pred.shape[1]
                    #     D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
                    #     target_one_hot = 2*torch.nn.functional.one_hot(target, num_classes = n_classes).type(torch.cuda.FloatTensor) - 1.0
                    #     diag_vector = 2/np.sqrt(np.pi)*(torch.exp(-(-target_one_hot*f_bar)**2))
                    #     D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
                    else:
                        # TODO Use autodiff do compute second derivative for given loss function
                        # Use second formula from paper here? 
                        D = 1.0

                    # f_loss = self.loss_function(f_bar, target).mean()
                    #sum_losses = f_loss
                    sum_losses = None
                    for i, pred in enumerate(base_preds):
                        diff = pred - f_bar.detach() #.detach()
                        covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
                        
                        reg = 1.0/self.n_estimators * 0.5 * covar
                        i_loss = self.loss_function(pred, target)
                        if sample_weight is not None:
                            i_loss = weights * i_loss
                            
                        ensemble_losses[i] += i_loss.sum().item()
                        ensemble_reg[i] += reg.sum().item()

                        reg_loss = i_loss.mean() - self.l_reg * reg.mean()
                        if sum_losses is not None:
                            sum_losses += reg_loss
                        else:
                            sum_losses = reg_loss

                    sum_losses.backward()
                    optimizer.step()

                    desc = '[{}/{}] loss {:2.4f} acc {:2.3f} bias {:2.4f} var {:2.4f} avg train acc {:4.3f} min {:4.3f} max {:4.3f}'.format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt, 
                        np.mean(ensemble_losses) / example_cnt, 
                        np.mean(ensemble_reg) / example_cnt, 
                        100. * np.mean(avg_n_correct)/example_cnt,
                        100. * min(avg_n_correct)/example_cnt,
                        100. * max(avg_n_correct)/example_cnt
                    )
                    pbar.set_description(desc)
            
                if self.x_test is not None and self.eval_test is not None and self.eval_test > 0 and epoch % self.eval_test == 0:
                    output = apply_in_batches(self, self.x_test, batch_size=self.batch_size)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    desc = '[{}/{}] loss {:2.4f} acc {:2.3f} bias {:2.3f} var {:2.4f} avg train acc {:4.3f} min {:4.3f} max {:4.3f} test acc {:4.3f}'.format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt, 
                        np.mean(ensemble_losses) / example_cnt, 
                        np.mean(ensemble_reg) / example_cnt, 
                        100. * np.mean(avg_n_correct)/example_cnt,
                        100. * min(avg_n_correct)/example_cnt,
                        100. * max(avg_n_correct)/example_cnt,
                        accuracy_test
                    )

                    pbar.set_description(desc)

            scheduler.step()
            # for s in schedulers:
            #     s.step()
            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                if self.x_test is not None and self.eval_test is not None and self.eval_test > 0 and epoch % self.eval_test == 0:
                    output = apply_in_batches(self, self.x_test)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    out_str = "{},{},{},{},{},{}".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0*n_correct/example_cnt,
                        100. * np.mean(avg_n_correct)/example_cnt,
                        accuracy_test,
                        np.mean(all_accuracy_test)
                    )
                else:
                    out_str = "{},{},{},{}".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0*n_correct/example_cnt,
                        100. * np.mean(avg_n_correct)/example_cnt,
                    )
                
                for l in ensemble_losses:
                    out_str += "," + str(l/example_cnt)
                for r in ensemble_reg:
                    out_str += "," + str(r/example_cnt)

                outfile.write(out_str + "\n")


# # TODO TAKE CARE OF HESSIAN HERE. CURRENTLY ASSSUME H = 1
                    # stacked_diff = torch.stack(diffs, dim=1)
                    # # print("STACKED_DIFF: ", stacked_diff.shape)
                    # covars = []
                    # for d in stacked_diff:
                    #     C = cov(d, bias=True, rowvar=False)
                    #     covars.append(C)
                    # covar = torch.stack(covars, dim=0) 
                    # trcovar = torch.diagonal(covar, dim1=-2, dim2=-1).sum(-1) 

                    # print("COVARS:", stacked_covars.shape)

                    # s_loss = loss.sum()
                    # grads,  = torch.autograd.grad(s_loss, f_bar, create_graph=True)
                    # print("GRADS:", grads.shape)

                    # hessians = []
                    # for i, (grad, p) in enumerate(zip(grads, f_bar)):
                    #     grad = grad.reshape(-1)
                    #     d = len(grad)
                    #     dg = torch.zeros((d, d))

                    #     for j, g in enumerate(grad):
                    #         g2 = torch.autograd.grad(g, p, create_graph=True)[0].view(-1)
                    #         dg[j] = g2

                    #     hessians.append(dg)

                    # H = torch.stack(hessians, dim=1)
                    # print("HESS:", H.shape)

                    # # def f_summed():
                    # #     return self.loss_function(target, )
                    # # H = torch.autograd.functional.hessian(self.loss_function, f_bar)
                    # # print("HESS:", H.shape)
                    # asdf
                    # hessians = []
                    # #l_sum = loss.sum()
                    # for cur_pred, cur_loss in zip(f_bar, loss):
                    #     print("CP: ", cur_pred.shape)
                    #     print("CL: ", cur_loss.shape)
                    #     H = hessian(cur_loss, cur_pred)
                    #     hessians.append(H)
                    # stacked_hessians = torch.stack(hessians, dim=0) 

                    # # What happend with the batch dimension here?!
                    # print("HESSAINS:", stacked_hessians.shape)
                    # print("COVARS:", stacked_covars.shape)
                    # # print(D)
                    # asdf