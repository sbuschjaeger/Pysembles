#!/usr/bin/env python3

from tqdm import tqdm
import numpy as np
import torch

from torch import nn
from torch.autograd import Variable

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Utils import apply_in_batches, TransformTensorDataset
from .Models import SKLearnModel
from .Models import StagedEnsemble
from .BinarisedNeuralNetworks import BinaryConv2d, BinaryLinear

import copy

class RandomFeature(nn.Module):
    def __init__(self, dimension, prob_one = 0.75):
        super().__init__()
        # low = 0 (inclusive), high = 1 (exclusive) -> use 2 here
        #self.random_projection = torch.randint(0,2,dimension).cuda()
        
        self.prob_one = prob_one
        self.random_projection = torch.zeros(size=dimension).cuda()
        self.random_projection.bernoulli_(self.prob_one)
        self.random_projection.requires_grad = False

    def forward(self, x):
        # TODO REMOVE squeeze ?
        # TODO MAKE SURE SIZE IS CORRECT?
        x_tmp = x.squeeze(1)
        rnd = torch.cat(x.shape[0]*[self.random_projection.squeeze(1)])
        #print("rnd: ", rnd.shape)
        tmp2 = torch.bmm(x_tmp,rnd.transpose(1,2))
        tmp2 = tmp2.unsqueeze(1)
        #print(x.shape)
        #print(tmp2.shape)
        return tmp2

        #rnd = self.random_projection.squeeze(1)
        # print(self.random_projection.shape)
        # print(x.shape)
        #print("X:", x_tmp.shape)
        #print("RND:", rnd.transpose(1,2).shape)
        #tmp2 = torch.bmm(x_tmp,rnd.view(rnd.shape[1],rnd.shape[2],rnd.shape[0]))

        # tmp = x * self.random_projection
        # print("TMP: ", tmp)
        # print("TMP2: ", tmp2)
        # print(tmp.shape)
        # print(tmp2.shape)
        # asdf
        # return tmp

class BaggingClassifier(StagedEnsemble):
    def __init__(self, n_estimators = 5, bootstrap = True, frac_examples = 1.0, freeze_layers = None, random_features = False, train_method = "fast", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.frac_samples = frac_examples
        self.bootstrap = bootstrap
        self.freeze_layers = freeze_layers
        self.args = args
        self.kwargs = kwargs
        self.random_features = random_features
        self.train_method = train_method

    def fit(self, X, y): 
        self.classes_ = unique_labels(y)
        if self.pipeline:
            X = self.pipeline.fit_transform(X)

        self.X_ = X
        self.y_ = y

        self.estimators_ = nn.ModuleList([
            SKLearnModel(training_csv="training_{}.csv".format(i), *self.args, **self.kwargs) for i in range(self.n_estimators)
        ])
        
        if self.freeze_layers is not None:
            for e in self.estimators_:
                for i, l in enumerate(e.layers_[:self.freeze_layers]):
                    # print("Layer {} which is {} is now frozen".format(i,l))
                    #if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, BinaryConv2d, nn.Linear, BinaryLinear)):
                    for p in l.parameters():
                        p.requires_grad = False
        
        if self.random_features:
            for e in self.estimators_:
                # TODO Make sure that if there is a transformer in the base estimator, that the shape is also correct here
                combined = [RandomFeature(dimension = X[0].shape), *e.layers_]
                e.layers_ = nn.Sequential( *combined)
                # print(e.layers_)
                # asdf

        # Check if we use the "fast" method for training. "Fast" copies the entire dataset multiple times and 
        # calls the forward method for each batch manually for each base model. 
        # The other method is the "regular" bagging-style training approach in which we simply call each
        # fit method individually. This trains one model after another which does not fully utilize 
        # the GPU for smaller base models, but might be faster if the base models are already quite large
        if self.train_method != "fast":
            for idx, est in enumerate(self.estimators_):
                if self.seed is not None:
                    np.random.seed(self.seed + idx)

                idx_array = [i for i in range(len(y))]
                idx_sampled = np.random.choice(
                    idx_array, 
                    size=int(self.frac_samples*len(idx_array)), 
                    replace=self.bootstrap
                )
                X_sampled = X[idx_sampled,] 
                y_sampled = y[idx_sampled]
                est.fit(X_sampled, y_sampled)
        else:
            # We follow the idea of Oza and Russell in "Online Bagging and Boosting" (https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf)
            # which suggest to weight each example with w ~ Possion(1) for each base-learner

            x_tensor = torch.tensor(X)
            y_tensor = torch.tensor(y)  
            w_tensor = torch.poisson(torch.ones(size=(len(y), self.n_estimators)))
            #w_tensor = torch.tensor(weights)  
            y_tensor = y_tensor.type(torch.LongTensor) 

            data = TransformTensorDataset(x_tensor,y_tensor,w_tensor,transform=self.transformer)
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
                    o_str = "epoch,loss,train-accuracy,avg-loss,avg-train-accuracy,test-loss,test-accuracy,avg-test-loss,avg-test-accuracy"
                else:
                    o_str = "epoch,loss,train-accuracy,avg-loss,avg-train-accuracy"
                outfile.write(o_str + "\n")
            
            self.train()
            for epoch in range(self.epochs):
                n_correct = 0
                ensemble_accuracy = [0 for _ in range(self.n_estimators)]
                ensemble_losses = [0 for _ in range(self.n_estimators)]
                example_cnt = 0
                batch_cnt = 0
                total_loss = 0

                with tqdm(total=len(train_loader.dataset), ncols=145, disable= not self.verbose) as pbar:
                    for batch_idx, batch in enumerate(train_loader):
                        data = batch[0]
                        target = batch[1]
                        weights = batch[2]
                        data, target = data.cuda(), target.cuda()
                        data, target = Variable(data), Variable(target)
                        
                        optimizer.zero_grad()
                        f_bar, base_preds = self.forward_with_base(data)

                        for i, pred in enumerate(base_preds):
                            acc = (pred.argmax(1) == target).type(torch.cuda.FloatTensor)
                            ensemble_accuracy[i] += acc.sum().item()

                        accuracy = (f_bar.argmax(1) == target).type(torch.cuda.FloatTensor)
                        loss = self.loss_function(f_bar, target)
                        total_loss += loss.sum().item()
                        n_correct += accuracy.sum().item()
                        
                        pbar.update(data.shape[0])
                        example_cnt += data.shape[0]
                        batch_cnt += 1

                        sum_losses = None
                        for i, pred in enumerate(base_preds):
                            # TODO: PyTorch copyies the weight vectgor if we use weights[:,i] to index
                            #       a specific row. Maybe we should re-factor this?
                            i_loss = self.loss_function(pred, target) * weights[:,i].cuda()
                            s_loss = i_loss.mean()
                            ensemble_losses[i] += s_loss.item()

                            if sum_losses is not None:
                                sum_losses += s_loss
                            else:
                                sum_losses = s_loss

                        sum_losses.backward()
                        optimizer.step()

                        desc = '[{}/{}] loss {:2.4f} acc {:2.3f} avg loss {:2.3f} avg acc {:2.3f}'.format(
                            epoch, 
                            self.epochs-1, 
                            total_loss/example_cnt, 
                            100.0 * n_correct/example_cnt, 
                            np.mean(ensemble_losses) / example_cnt, 
                            100.0 * np.mean(ensemble_accuracy) / example_cnt, 
                        )
                        pbar.set_description(desc)
                
                    if self.x_test is not None and self.eval_test is not None and self.eval_test > 0 and epoch % self.eval_test == 0:
                        output = apply_in_batches(self, self.x_test, batch_size=self.batch_size)
                        accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                        ensemble_accuracy_test = []
                        for e in self.estimators_:
                            e_output = apply_in_batches(e, self.x_test)
                            e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                            ensemble_accuracy_test.append(e_acc)

                        desc = '[{}/{}] loss {:2.4f} acc {:2.3f} avg loss {:2.3f} avg acc {:2.3f} test acc {:2.3f} avg test acc {:2.3f}'.format(
                            epoch, 
                            self.epochs-1, 
                            total_loss/example_cnt, 
                            100.0 * n_correct/example_cnt, 
                            np.mean(ensemble_losses) / example_cnt, 
                            100.0 * np.mean(ensemble_accuracy) / example_cnt, 
                            accuracy_test,
                            np.mean(ensemble_accuracy_test)
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

                    ensemble_accuracy_test = []
                    for e in self.estimators_:
                        e_output = apply_in_batches(e, self.x_test)
                        e_acc = accuracy_score(np.argmax(e_output, axis=1),self.y_test)*100.0
                        ensemble_accuracy_test.append(e_acc)

                    out_str = "{},{},{},{},{},{},{}".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0 * n_correct/example_cnt, 
                        np.mean(ensemble_losses) / example_cnt, 
                        100.0 * np.mean(ensemble_accuracy) / example_cnt, 
                        accuracy_test,
                        np.mean(ensemble_accuracy_test)
                    )
                else:
                    out_str = "{},{},{},{},{}".format(
                        epoch, 
                        total_loss/example_cnt, 
                        100.0 * n_correct/example_cnt,
                        np.mean(ensemble_losses) / example_cnt, 
                        100.0 * np.mean(ensemble_accuracy) / example_cnt, 
                    )
                
                outfile.write(out_str + "\n")
