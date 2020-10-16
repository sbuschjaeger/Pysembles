import os
import random
import copy
import json
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Utils import store_model, TransformTensorDataset, apply_in_batches

class SKLearnBaseModel(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer, scheduler, loss_function, 
                 base_estimator, 
                 training_file="training.jsonl",
                 transformer = None,
                 pipeline = None,
                 seed = None,
                 verbose = True, out_path = None, 
                 x_test = None, y_test = None, 
                 eval_test = 5,
                 store_on_eval = False) :
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
        self.store_on_eval = store_on_eval
        self.training_file = training_file

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # if you are using GPU
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def store(self, out_path, dim, name="model"):
        shallow_copy = copy.copy(self)
        shallow_copy.X_ = np.array(1)
        shallow_copy.y_ = np.array(1)
        shallow_copy.base_estimator = None
        shallow_copy.x_test = None
        shallow_copy.y_test = None
        torch.save(shallow_copy, os.path.join(out_path, name + ".pickle"))
        store_model(self, "{}/{}.onnx".format(out_path, name), dim, verbose=self.verbose)

    def predict_proba(self, X, eval_mode=True):
        # print("pred proba", X.shape)
        check_is_fitted(self, ['X_', 'y_'])
        before_eval = self.training
        
        if eval_mode:
            self.eval()
        else:
            self.train()

        self.cuda()
        y_pred = None
        with torch.no_grad(): 
            if self.pipeline:
                X = self.pipeline.transform(X)
            
            x_tensor = torch.tensor(X)
            
            # At some point during development we had problems with inconsistend 
            # data types and data interpretations and therefore we introduced a transformer for
            # both, testing and training. It seems that PyTorch got this sorted now and 
            # thus we do not need it anymore?
            # if hasattr(model, "transformer") and model.transformer is not None:
            #     test_transformer =  None 
            #     # transforms.Compose([
            #     #     transforms.ToPILImage(),
            #     #     transforms.ToTensor() 
            #     # ])
            # else:
            #     test_transformer = None
            test_transformer = None
            dataset = TransformTensorDataset(x_tensor,transform=test_transformer)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size)
            for data in train_loader:
                data = data.cuda()
                pred = self(data)
                pred = pred.cpu().detach().numpy()
                if y_pred is None:
                    y_pred = pred
                else:
                    y_pred = np.concatenate( (y_pred, pred), axis=0 )
            return y_pred

        self.train(before_eval)
        return y_pred

    def predict(self, X, eval_mode=True):
        # print("pred", X.shape)
        pred = self.predict_proba(X, eval_mode)
        return np.argmax(pred, axis=1)

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        if self.pipeline:
            X = self.pipeline.fit_transform(X)

        x_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        y_tensor = y_tensor.type(torch.LongTensor) 

        if sample_weight is not None:
            #sample_weight = len(y)*sample_weight/np.sum(sample_weight)
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
            outfile = open(self.out_path + "/" + self.training_file, "w", 1)

        for epoch in range(self.epochs):
            metrics = {}
            example_cnt = 0

            with tqdm(total=len(train_loader.dataset), ncols=150, disable = not self.verbose) as pbar:
                for batch in train_loader:
                    data = batch[0]
                    target = batch[1]
                    data, target = data.cuda(), target.cuda()
                    data, target = Variable(data), Variable(target)
                    example_cnt += data.shape[0]

                    if sample_weight is not None:
                        weights = batch[2]
                        weights = weights.cuda()
                        weights = Variable(weights)
                    else:
                        weights = None

                    optimizer.zero_grad()

                    # We assume that prepare_backward computes the appropriate loss and possible some statistics 
                    # the user wants to store / output. To do so, prepare_backward should return a dictionary with 
                    # three fields. An example is given below. Note that the prediction / loss / metrics should be
                    # given for each individual example in the batch. 
                    #    !!!! Do not reduce / sum / mean the loss etc manually !!!!
                    # This is done afterwards in this code. 
                    #
                    # d = {
                    #     "prediction" : self(data), 
                    #     "backward" : self.loss_function(self(data), target), 
                    #     "metrics" :
                    #     {
                    #         "loss" : self.loss_function(self(data), target),
                    #         "accuracy" : 100.0*(self(data).argmax(1) == target).type(torch.cuda.FloatTensor)
                    #     } 
                    # }
                    backward = self.prepare_backward(data, target, weights)
                    
                    for key,val in backward["metrics"].items():
                        metrics[key] = metrics.get(key,0) + val.sum().item()
                    
                    backward["backward"].mean().backward()
                    optimizer.step()

                    mstr = ""
                    for key,val in metrics.items():
                        mstr += "{} {:2.4f} ".format(key, val / example_cnt)

                    pbar.update(data.shape[0])
                    desc = '[{}/{}] {}'.format(epoch, self.epochs-1, mstr)
                    pbar.set_description(desc)

                if scheduler is not None:
                    scheduler.step()

                torch.cuda.empty_cache()
                
                if self.out_path is not None:
                    out_dict = {}
                    
                    mstr = ""
                    for key,val in metrics.items():
                        out_dict["train_" + key] = val / example_cnt
                        mstr += "{} {:2.4f} ".format(key, val / example_cnt)

                    if self.x_test is not None and self.eval_test is not None and self.eval_test > 0 and epoch % self.eval_test == 0:
                        if self.store_on_eval:
                            torch.save(self.state_dict(), os.path.join(self.out_path, 'model_{}.checkpoint'.format(epoch)))

                        # This is basically a different version of apply_in_batches but using the "new" prepare_backward interface
                        # for evaluating the test data. Maybe we should refactor this at some point and / or apply_in_batches
                        # is not really needed anymore as its own function?
                        # TODO Check if refactoring might be interestring here
                        self.eval()

                        test_metrics = {}
                        x_tensor_test = torch.tensor(self.x_test)
                        y_tensor_test = torch.tensor(self.y_test)

                        dataset = TransformTensorDataset(x_tensor_test,y_tensor_test, transform=None)
                        test_loader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size)
                        
                        for batch in test_loader:
                            test_data = batch[0]
                            test_target = batch[1]
                            test_data, test_target = test_data.cuda(), test_target.cuda()
                            test_data, test_target = Variable(test_data), Variable(test_target)
                            with torch.no_grad():
                                backward = self.prepare_backward(test_data, test_target)

                            for key,val in backward["metrics"].items():
                                test_metrics[key] = test_metrics.get(key,0) + val.sum().item()

                        self.train()
                        for key,val in test_metrics.items():
                            out_dict["test_" + key] = val / len(self.y_test)
                            mstr += "test {} {:2.4f} ".format(key, val / len(self.y_test))
                    
                    desc = '[{}/{}] {}'.format(epoch, self.epochs-1, mstr)
                    pbar.set_description(desc)
                    
                    out_dict["epoch"] = epoch
                    out_file_content = json.dumps(out_dict, sort_keys=True) + "\n"
                    outfile.write(out_file_content)

class SKEnsemble(SKLearnBaseModel):
    def __init__(self, n_estimators = 5, combination_type = "average", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combination_type = combination_type
        self.n_estimators = n_estimators
        
        assert self.n_estimators > 0, "Your ensemble should have at-least one member, but self.n_estimators was {}".format(self.n_estimators)
        assert self.combination_type  in ('average', 'softmax', 'best'), "Combination type must be one of ('average', 'softmax', 'best')"

    def forward(self, X):
        return self.forward_with_base(X)[0]

    # @torch.jit.script
    def forward_with_base(self, X):
        # base_preds = []
        # for i in torch.range(start=0,end=5):
        #     base_preds.append(self.estimators_[i](X)) 

        # # base_preds = [self.estimators_[i](X) for i in torch.range(self.n_estimators)] #self.n_estimators
        base_preds = [e(X) for e in self.estimators_] #self.n_estimators
        if self.combination_type == "average":
            pred_combined = 1.0/self.n_estimators*torch.sum(torch.stack(base_preds, dim=1),dim=1)
        elif self.combination_type == "softmax":
            pred_combined = 1.0/self.n_estimators*torch.sum(torch.stack(base_preds, dim=1),dim=1)
            pred_combined = nn.functional.softmax(pred_combined,dim=1)
        else:   
            pred_combined, _ = torch.stack(base_preds, dim=1).max(dim=1)
        
        return pred_combined, base_preds

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.base_estimator()
        
    def prepare_backward(self, data, target, weights = None):
        output = self(data)
        loss = self.loss_function(output, target)

        if weights is not None:
            loss = loss * weights

        d = {
            "prediction" : output, 
            "backward" : loss, 
            "metrics" :
            {
                "loss" : loss.detach(),
                "accuracy" : 100.0*(output.argmax(1) == target).type(torch.cuda.FloatTensor)
            } 
            
        }
        return d

    def forward(self, x):
        try:
            return self.model(x)
        except Exception as e:
            for l in self.model.children():
                # print(l)
                x = l(x)
                print(x.shape)
            raise e
        # return self.layers_(x)
