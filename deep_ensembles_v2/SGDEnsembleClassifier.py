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
from .Utils import apply_in_batches, cov, weighted_mse_loss, weighted_squared_hinge_loss

class SGDEnsembleClassifier(SKEnsemble):
    def __init__(self, n_estimators = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators

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
                o_str = "epoch,loss,train-accuracy,avg-train-accuracy,test-accuracy,avg-test-accuracy"
            else:
                o_str = "epoch,loss,train-accuracy,avg-train-accuracy"
            outfile.write(o_str + "\n")
        
        self.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_correct = 0
            avg_n_correct = [0 for _ in range(self.n_estimators)]
            example_cnt = 0
            batch_cnt = 0

            with tqdm(total=len(train_loader.dataset), ncols=165, disable= not self.verbose) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)
                    
                    optimizer.zero_grad()
                    f_bar, base_preds = self.forward_with_base(data)
                    
                    for i, pred in enumerate(base_preds):
                        acc = (pred.argmax(1) == target).type(torch.cuda.FloatTensor)
                        avg_n_correct[i] +=acc.sum().item()
                        # avg_n_correct += acc.sum().item()

                    accuracy = (f_bar.argmax(1) == target).type(torch.cuda.FloatTensor)
                    loss = self.loss_function(f_bar, target)
                    total_loss += loss.sum().item()
                    n_correct += accuracy.sum().item()
                    
                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]
                    batch_cnt += 1

                    loss = loss.mean() 

                    loss.backward()
                    optimizer.step()

                    desc = "[{}/{}] loss {:4.3f} acc {:4.2f} avg acc {:4.2f} min {:4.2f} max {:4.2f}".format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        100. * np.mean(avg_n_correct)/(example_cnt),
                        100. * min(avg_n_correct)/example_cnt,
                        100. * max(avg_n_correct)/example_cnt
                    )

                    pbar.set_description(desc)
                
                if self.x_test is not None and self.eval_test is not None and self.eval_test > 0:
                    output = apply_in_batches(self, self.x_test, batch_size = self.batch_size)
                    accuracy_test_apply = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    output_proba = self.predict_proba(self.x_test)
                    accuracy_test_proba = accuracy_score(np.argmax(output_proba, axis=1),self.y_test)*100.0

                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test, batch_size = self.batch_size)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    desc = "[{}/{}] loss {:4.3f} acc {:4.2f} avg acc {:4.2f} min {:4.2f} max {:4.2f} test acc {:4.3f} avg test acc {:4.3f} test acc proba {:4.3f}".format(
                        epoch, 
                        self.epochs-1, 
                        total_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        100. * np.mean(avg_n_correct)/(example_cnt),
                        100. *min(avg_n_correct)/example_cnt,
                        100. *max(avg_n_correct)/example_cnt,
                        accuracy_test_apply,
                        np.mean(all_accuracy_test), 
                        accuracy_test_proba
                    )

                    pbar.set_description(desc)
            
            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                if self.x_test is not None and self.eval_test is not None and self.eval_test > 0:
                    output= self.predict_proba(self.x_test)
                    # output = apply_in_batches(self, self.x_test)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0
                    all_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test, batch_size = self.batch_size)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        all_accuracy_test.append(e_acc)

                    o_str = "{},{},{},{},{},{},{},{}\n".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0 * n_correct/example_cnt, 
                        100. * np.mean(avg_n_correct)/(example_cnt),
                        100. *min(avg_n_correct)/example_cnt,
                        100. *max(avg_n_correct)/example_cnt,
                        accuracy_test,
                        np.mean(all_accuracy_test)
                    )
                else:
                    o_str = "{},{},{},{},{},{}\n".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0 * n_correct/example_cnt, 
                        100. * np.mean(avg_n_correct)/(example_cnt),
                        100. *min(avg_n_correct)/example_cnt,
                        100. *max(avg_n_correct)/example_cnt,
                    )
                outfile.write(o_str)
                if epoch % 10 == 0:
                    torch.save(self.state_dict(), os.path.join(self.out_path, 'model_{}.checkpoint'.format(epoch)))
                