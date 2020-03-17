import os
import random
import copy
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from Utils import apply_in_batches, TransformTensorDataset

class SKLearnBaseModel(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer, scheduler, loss_function, 
                 base_estimator, 
                 transformer = None,
                 pipeline = None,
                 seed = None,
                 verbose = True, out_path = None, 
                 x_test = None, y_test = None, eval_test = 5) :
        super().__init__()
        
        if optimizer is not None:
            optimizer_copy = copy.deepcopy(optimizer)
            self.batch_size = optimizer_copy.pop("batch_size")
            self.epochs = optimizer_copy.pop("epochs")
            self.optimizer_method = optimizer_copy.pop("method")
            self.optimizer = optimizer_copy
        else:
            self.optimizer = None

        if scheduler is not None:
            scheduler_copy = copy.deepcopy(scheduler)

            self.scheduler_method = scheduler_copy.pop("method")
            self.scheduler = scheduler_copy

        else:
            self.scheduler = None
            
        self.base_estimator = base_estimator
        self.loss_function = loss_function
        self.transformer = transformer
        self.pipeline = pipeline
        self.verbose = verbose
        self.out_path = out_path
        self.x_test = x_test
        self.y_test = y_test
        self.seed = seed
        self.eval_test = eval_test
        self.layers_ = self.base_estimator()
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # if you are using GPU
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def predict_proba(self, X):
        # print("pred proba", X.shape)
        check_is_fitted(self, ['X_', 'y_'])
        before_eval = self.training
        self.eval()
        self.cuda()
        with torch.no_grad(): 
            ret_val = apply_in_batches(self, X, batch_size = self.batch_size)

        self.train(before_eval)
        return ret_val

    def predict(self, X):
        # print("pred", X.shape)
        pred = self.predict_proba(X)
        return np.argmax(pred, axis=1)

class SKEnsemble(SKLearnBaseModel):
    def forward(self, X):
        return self.forward_with_base(X)[0]

    def forward_with_base(self, X):
        base_preds = [self.estimators_[i](X) for i in range(self.n_estimators)]
        pred_combined = 1.0/self.n_estimators*torch.sum(torch.stack(base_preds, dim=1),dim=1)
        return pred_combined, base_preds

class StagedEnsemble(SKEnsemble):
    # Assumes self.estimators_ and self.estimator_weights_ exists
    def staged_predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])

        if not hasattr(self, 'n_estimators'):
            errormsg = '''staged_predict_proba was called on SKLearnBaseModel without its subclass {}
                          beeing an ensemble (n_estimators attribute not found)!'''.format(self.__class__.__name__)
            raise AttributeError(errormsg)

        self.eval()

        with torch.no_grad():
            all_pred = None
            for i, est in enumerate(self.estimators_):
                y_pred = apply_in_batches(est, X, batch_size = self.batch_size)
                
                if all_pred is None:
                    all_pred = 1.0/self.n_estimators*y_pred
                else:
                    all_pred = all_pred + 1.0/self.n_estimators*y_pred

                yield all_pred*self.n_estimators/(i+1)

class SKLearnModel(SKLearnBaseModel):
    def __init__(self, training_csv="training.csv", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_csv = training_csv

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        if self.pipeline:
            X = self.pipeline.fit_transform(X)
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
                o_str = "epoch,train-loss,train-accuracy,test-loss,test-accuracy"
            else:
                o_str = "epoch,train-loss,train-accuracy"

            outfile.write(o_str + "\n")

        for epoch in range(self.epochs):
            epoch_loss = 0
            n_correct = 0
            example_cnt = 0
            batch_cnt = 0
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
                    # print("")
                    # print(self.layers_[-3].running_var)
                    # print(self.layers_[-3].running_mean)
                    # print("")
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt
                    )
                    pbar.set_description(desc)
            
                if self.x_test is not None and epoch % self.eval_test == 0:
                    pred_proba = self.predict_proba(self.x_test)
                    pred_tensor = torch.tensor(pred_proba).cuda()
                    y_test_tensor = torch.tensor(self.y_test).cuda()
                    test_loss = self.loss_function(pred_tensor, y_test_tensor).mean().item()
                    accuracy_test = accuracy_score(self.y_test, np.argmax(pred_proba, axis=1))*100.0  
                    
                    desc = '[{}/{}] loss {:2.4f} train acc {:2.4f} test loss/acc {:2.4f}/{:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        test_loss,
                        accuracy_test
                    )
                    pbar.set_description(desc)

            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                avg_loss = epoch_loss/example_cnt
                accuracy = 100.0*n_correct/example_cnt

                if self.x_test is not None:
                    if epoch % self.eval_test != 0:
                        accuracy_test = "-"
                        test_loss = "-"

                    outfile.write("{},{},{},{},{}\n".format(epoch, avg_loss, accuracy, test_loss, accuracy_test))
                else:
                    outfile.write("{},{},{}\n".format(epoch, avg_loss, accuracy))
        
    def forward(self, x):
        return self.layers_(x)


    def predict_proba(self, X):
        if self.pipeline:
            return super().predict_proba(self.pipeline.transform(X))
        return super().predict_proba(X)
