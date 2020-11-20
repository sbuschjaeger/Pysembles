import os
import random
import copy
import json
import types
import torch
from torch import nn
from torch.autograd import Variable
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from pysembles.Metrics import avg_accurcay,diversity,avg_loss,loss
from pysembles.Utils import pytorch_total_params, apply_in_batches, TransformTensorDataset

from .Utils import store_model, TransformTensorDataset, apply_in_batches

class BaseModel(nn.Module):
    def __init__(self, 
                    optimizer, 
                    scheduler, 
                    loss_function, 
                    base_estimator, 
                    training_file="training.jsonl",
                    seed = None,
                    verbose = True, 
                    out_path = None, 
                    test_data = None,
                    eval_every = 5,
                    store_every = None,
                    device = "cuda",
                    loader = None,
                    use_amp = False,
                    *args,**kwargs
                ) :
        super().__init__()
        
        if isinstance(base_estimator, types.LambdaType) and base_estimator.__name__ == "<lambda>":
            print("Warning: base_estimator is a lambda function in Models.py - This is fine, unless you want to store checkpoints of your model. This will likely fail since unnamed functions cannot be pickled. Consider naming it.")

        if optimizer is not None:
            optimizer_copy = copy.deepcopy(optimizer)
            self.optimizer_method = optimizer_copy.pop("method")
            if "epochs" in optimizer_copy:
                self.epochs = optimizer_copy.pop("epochs")
            else:
                self.epochs = 1

            self.optimizer_cfg = optimizer_copy
        else:
            self.optimizer_cfg = None

        if scheduler is not None:
            scheduler_copy = copy.deepcopy(scheduler)
            self.scheduler_method = scheduler_copy.pop("method")
            self.scheduler_cfg = scheduler_copy
        else:
            self.scheduler_cfg = None
            
        if loader is not None:
            self.loader_cfg = loader
        else:
            self.loader_cfg =  {'num_workers': 1, 'pin_memory': True, 'batch_size':128} 

        self.base_estimator = base_estimator
        self.loss_function = loss_function
        self.verbose = verbose
        self.out_path = out_path
        self.test_data = test_data
        self.seed = seed
        self.eval_every = eval_every
        self.store_every = store_every
        self.training_file = training_file
        self.cur_epoch = 0
        self.resume_from_checkpoint = False
        self.device = device
        self.use_amp = use_amp
        
        if self.seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # if you are using GPU
            if self.device != "cpu":
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    def get_float_type(self):
        if self.device == "cpu":
            return torch.FloatTensor
        else:
            return torch.cuda.FloatTensor

    def restore_state(self,checkpoint):
        self.optimizer_method = checkpoint["optimizer_method"]
        self.optimizer_cfg = checkpoint["optimizer_cfg"]
        self.scheduler_method = checkpoint["scheduler_method"]
        self.scheduler_cfg = checkpoint["scheduler_cfg"]
        self.loader_cfg = checkpoint["loader_cfg"]
        self.scheduler = checkpoint["scheduler"]
        self.base_estimator = checkpoint["base_estimator"]
        self.loss_function = checkpoint["loss_function"]
        self.verbose = checkpoint["verbose"]
        self.out_path = checkpoint["out_path"]
        self.test_data = checkpoint["test_data"]
        self.seed = checkpoint["seed"]
        self.eval_every = checkpoint["eval_every"]
        self.store_every = checkpoint["store_every"]
        self.training_file = checkpoint["training_file"]
        self.cur_epoch = checkpoint["cur_epoch"]
        self.epochs = checkpoint["epochs"]
        self.resume_from_checkpoint = True
        self.device = checkpoint["device"]
        self.use_amp = checkpoint["use_amp"]
        self.scaler = GradScaler(enabled = self.use_amp)
        
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            # if you are using GPU
            if self.device != "cpu": 
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
        
        self.load_state_dict(checkpoint['state_dict'])

        # Load the model to the correct device _before_ we init the optimizer
        # https://github.com/pytorch/pytorch/issues/2830
        self.to(self.device)

        self.optimizer = self.optimizer_method(self.parameters(), **self.optimizer_cfg)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler_method is not None:
            self.scheduler = self.scheduler_method(self.optimizer, **self.scheduler_cfg)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            self.scheduler = None

        if self.loader_cfg is None:
            self.loader_cfg =  {'num_workers': 1, 'pin_memory': True, 'batch_size':128} 

    def restore_checkoint(self, path):
        # https://github.com/pytorch/pytorch/issues/2830
        checkpoint = torch.load(path, map_location = self.device)
        self.restore_state(checkpoint)

    def get_state(self):
        return {
            "optimizer_method" : self.optimizer_method,
            "optimizer_cfg" : self.optimizer_cfg,
            "loader_cfg" : self.loader_cfg,
            "scheduler_method" : self.scheduler_method,
            "scheduler_cfg" : self.scheduler_cfg,
            "scheduler" : self.scheduler,
            "base_estimator" : self.base_estimator,
            "loss_function" : self.loss_function,
            "verbose" : self.verbose,
            "out_path" : self.out_path,
            "test_data" : self.test_data,
            "seed" : self.seed,
            "device" : self.device,
            "eval_every" : self.eval_every,
            "store_every" : self.store_every,
            "training_file" : self.training_file,
            'cur_epoch': self.cur_epoch,
            'epochs': self.epochs, 
            'use_amp': self.use_amp, 
            'state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': None if not self.scheduler else self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }

    def store_checkpoint(self):
        state = self.get_state()
        torch.save(state, os.path.join(self.out_path, 'model_{}.tar'.format(self.cur_epoch)))

    def fit(self, data):
        if not self.resume_from_checkpoint:
            self.optimizer = self.optimizer_method(self.parameters(), **self.optimizer_cfg)
            
            if self.scheduler_method is not None:
                self.scheduler = self.scheduler_method(self.optimizer, **self.scheduler_cfg)
            else:
                self.scheduler = None

            self.scaler = GradScaler(enabled=self.use_amp)

            if self.out_path is not None:
                outfile = open(self.out_path + "/" + self.training_file, "w", 1)
        else:
            if self.out_path is not None:
                outfile = open(self.out_path + "/" + self.training_file, "a", 1)

        train_loader = torch.utils.data.DataLoader(
            data,
            shuffle=True, 
            **self.loader_cfg
        )

        self.to(self.device)

        self.train()
        for epoch in range(self.cur_epoch, self.epochs):
            self.cur_epoch = epoch + 1
            metrics = {}
            example_cnt = 0

            with tqdm(total=len(train_loader.dataset), ncols=150, disable = not self.verbose) as pbar:
                self.batch_cnt = 0
                for batch in train_loader:
                    if len(batch) == 1:
                        data = batch
                    else:
                        data = batch[0]
                    
                    data = data.to(self.device)
                    data = Variable(data)

                    if len(batch) > 1:
                        target = batch[1]
                        target = target.to(self.device)
                        target = Variable(target)
                    else:
                        target = None

                    if len(batch) > 2:
                        weights = batch[2]
                        weights = weights.to(self.device)
                        weights = Variable(weights)
                    else:
                        weights = None

                    example_cnt += data.shape[0]

                    self.optimizer.zero_grad()

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
                    #         "accuracy" : 100.0*(self(data).argmax(1) == target).type(self.get_float_type())
                    #     } 
                    # }
                    with autocast(enabled = self.use_amp):
                        backward = self.prepare_backward(data, target, weights)
                        loss = backward["backward"].mean()

                    for key,val in backward["metrics"].items():
                        metrics[key] = metrics.get(key,0) + val.sum().item()
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    mstr = ""
                    for key,val in metrics.items():
                        mstr += "{} {:2.4f} ".format(key, val / example_cnt)

                    pbar.update(data.shape[0])
                    desc = '[{}/{}] {}'.format(epoch, self.epochs-1, mstr)
                    pbar.set_description(desc)
                    self.batch_cnt += 1

                if self.scheduler is not None:
                    self.scheduler.step()

                #torch.cuda.empty_cache()
                
                if self.out_path is not None:
                    out_dict = {}
                    
                    mstr = ""
                    for key,val in metrics.items():
                        out_dict["train_" + key] = val / example_cnt
                        mstr += "{} {:2.4f} ".format(key, val / example_cnt)

                    if self.store_every and self.store_every > 0 and (epoch % self.store_every) == 0:
                        self.store_checkpoint()

                    if self.test_data and self.eval_every and self.eval_every > 0 and (epoch % self.eval_every) == 0:
                        # This is basically a different version of apply_in_batches but using the "new" prepare_backward interface
                        # for evaluating the test data. Maybe we should refactor this at some point and / or apply_in_batches
                        # is not really needed anymore as its own function?
                        # TODO Check if refactoring might be interestring here
                        self.eval()

                        test_metrics = {}
                        test_loader = torch.utils.data.DataLoader(self.test_data, **self.loader_cfg)
                        
                        for batch in test_loader:
                            test_data = batch[0]
                            test_target = batch[1]
                            test_data, test_target = test_data.to(self.device), test_target.to(self.device)
                            test_data, test_target = Variable(test_data), Variable(test_target)
                            with torch.no_grad():
                                backward = self.prepare_backward(test_data, test_target)

                            for key,val in backward["metrics"].items():
                                test_metrics[key] = test_metrics.get(key,0) + val.sum().item()

                        self.train()
                        for key,val in test_metrics.items():
                            out_dict["test_" + key] = val / len(test_loader.dataset)
                            mstr += "test {} {:2.4f} ".format(key, val / len(test_loader.dataset))
                    
                    desc = '[{}/{}] {}'.format(epoch, self.epochs-1, mstr)
                    pbar.set_description(desc)
                    
                    out_dict["epoch"] = epoch
                    out_file_content = json.dumps(out_dict, sort_keys=True) + "\n"
                    outfile.write(out_file_content)

            if hasattr(train_loader.dataset, "end_of_epoch"):
                train_loader.dataset.end_of_epoch()

class Ensemble(BaseModel):
    def __init__(self, n_estimators = 5, combination_type = "average", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.combination_type = combination_type
        self.n_estimators = n_estimators
        
        assert self.n_estimators > 0, "Your ensemble should have at-least one member, but self.n_estimators was {}".format(self.n_estimators)
        assert self.combination_type  in ('average', 'softmax', 'best'), "Combination type must be one of ('average', 'softmax', 'best')"

    def restore_state(self, checkpoint):
        super().restore_state(checkpoint)
        self.combination_type = checkpoint["combination_type"]
        self.n_estimators = checkpoint["n_estimators"]

    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "combination_type":self.combination_type,
            "n_estimators":self.n_estimators
        }

    def forward(self, X):
        return self.forward_with_base(X)[0]

    # @torch.jit.script
    def forward_with_base(self, X):
        base_preds = [e(X) for e in self.estimators_] #self.n_estimators
        # Some ensemble methods may introduce / remove new models while optimization. Thus we cannot use
        # self.n_estimators which is the (maximum) number of models in the ensemble, but not the current one
        n_estimators = len(self.estimators_)
        if self.combination_type == "average":
            pred_combined = 1.0/n_estimators*torch.sum(torch.stack(base_preds, dim=1),dim=1)
        elif self.combination_type == "softmax":
            pred_combined = 1.0/n_estimators*torch.sum(torch.stack(base_preds, dim=1),dim=1)
            pred_combined = nn.functional.softmax(pred_combined,dim=1)
        else:   
            pred_combined, _ = torch.stack(base_preds, dim=1).max(dim=1)
        
        return pred_combined, base_preds

class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.base_estimator()

    def restore_state(self,checkpoint):
        super().restore_state(checkpoint)
        # I am not sure if model is part of the overall state_dict and thus properly loaded. 
        # Lets go the save route and store it as well
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "model_state_dict":self.model.state_dict(),
        }   

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
                "accuracy" : 100.0*(output.argmax(1) == target).type(self.get_float_type())
            } 
            
        }
        return d

    def forward(self, x):
        try:
            return self.model(x)
        except Exception as e:
            # This is just some random code which might help during debugging of the model, e.g. if the size of the linear layer does not
            # match the input after a flatten
            # Per convention we use "layers_" as a sequential which works nicely. If the model is more complex however, we simply
            # iterate over the children, which does not work as nicely if there is alreay a problem in a single child. 
            # TODO: We should enhance this, e.g. by recursivley executing children (lol, that sounds weird)
            if hasattr(self.model, "layers_"):
                layers = self.model.layers_
            else:
                layers = self.model.children()
                
            for l in layers:
                print("IN: ", x.shape)
                x = l(x)
                print("OUT: ", x.shape, " \n")
            raise e
        # return self.layers_(x)
