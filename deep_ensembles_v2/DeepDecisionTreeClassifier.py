#!/usr/bin/env python3

import os
import numpy as np
import torch
import random
import copy

from torch import nn
from torch.autograd import Variable

from tqdm import tqdm

from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

from Utils import apply_in_batches, TransformTensorDataset


def collect_all_path(node, cur_path, was_left, all_paths):
    # cur_path.append((was_left, node["model"]))
    cur_path.append((was_left, node))
    if node["is_leaf"]:
        all_paths.append(cur_path)
    else:
        collect_all_path(node["left"], cur_path.copy(), True, all_paths)
        collect_all_path(node["right"], cur_path.copy(), False, all_paths)

def build_tree(depth, leaf_estimator, split_estimator):
    root = {
        "is_leaf": False,
        "model": split_estimator(),
        "left":None,
        "right":None
    }
    all_nodes = [root]
    all_models = [root["model"]]

    to_expand = [root]
    for d in range(depth-1):
        new_expand = []
        for node in to_expand:
            left_node = {
                "is_leaf": False,
                "invert":False,
                "model": split_estimator(),
                "left":None,
                "right":None
            }
            right_node =  {
                "is_leaf": False,
                "invert":True,
                "model": split_estimator(),
                "left":None,
                "right":None
            }
            node["left"] = left_node
            new_expand.append(left_node)
            all_nodes.append(left_node)
            all_models.append(left_node["model"])

            node["right"] = right_node
            new_expand.append(right_node)
            all_nodes.append(right_node)
            all_models.append(right_node["model"])

        to_expand = new_expand 

    all_leafs = []
    for node in to_expand:
        left_node = {
            "is_leaf": True,
            "model": leaf_estimator(),
            "left":None,
            "right":None,
            "cnt":torch.tensor(0)
        }
        right_node =  {
            "is_leaf": True,
            "model": leaf_estimator(),
            "left":None,
            "right":None,
            "cnt":torch.tensor(0)
        }
        node["left"] = left_node
        all_nodes.append(left_node)
        all_models.append(left_node["model"])
        all_leafs.append(left_node)

        node["right"] = right_node
        all_nodes.append(right_node)
        all_models.append(right_node["model"])
        all_leafs.append(right_node)

    return all_nodes, all_models, all_leafs

class DeepDecisionTreeClassifier(nn.Module, BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer, scheduler, loss_function,  
                 depth, leaf_estimator, split_estimator,
                 transformer = None,
                 seed = None,
                 verbose = True, out_path = None, 
                 x_test = None, y_test = None) :
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
            
        self.depth = depth
        self.leaf_estimator = leaf_estimator
        self.split_estimator = split_estimator
        self.loss_function = loss_function
        self.transformer = transformer
        self.verbose = verbose
        self.out_path = out_path
        self.x_test = x_test
        self.y_test = y_test
        self.seed = seed
        self.training_csv = "training.csv"

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            # if you are using GPU
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def predict_proba(self, X):
        check_is_fitted(self, ['X_', 'y_'])
        before_eval = self.training
        self.eval()
        self.cuda()
        with torch.no_grad(): 
            ret_val = apply_in_batches(self, X, batch_size = self.batch_size)
            self.train(before_eval)
            return ret_val

    def predict(self, X):
        pred = self.predict_proba(X)
        return np.argmax(pred, axis=1)

    def fit(self, X, y, sample_weight = None): 
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        #self.layers_ = nn.Sequential(*[self.split_estimator(), self.leaf_estimator(), self.leaf_estimator()])

        self.layers_ = []
        n_inner = int((2**(self.depth+1) - 1)/2)
        n_leafs = 2**self.depth
        for i in range(n_inner):
             self.layers_.append(self.split_estimator())
        
        for i in range(n_leafs):
             self.layers_.append(self.leaf_estimator())
        self.layers_ = nn.Sequential(*self.layers_)
        
        cur_path = [[0]]
        for i in range(self.depth):
            tmp_path = []
            for p in cur_path:
                p1 = p.copy()
                p2 = p.copy()
                p1.append( 2*p[-1] + 1 )
                p2.append( 2*p[-1] + 2 )
                tmp_path.append(p1)
                tmp_path.append(p2)
            cur_path = tmp_path
        self.all_pathes = cur_path
        # print(self.layers_)
        #print(self.all_pathes)
        #asdf
        # self.all_nodes, all_models, self.all_leafs = build_tree(self.depth, self.leaf_estimator, self.split_estimator)
        # self.layers_ = nn.Sequential(*all_models)

        #self.all_paths = []
        #collect_all_path(self.all_nodes[0], [], False, self.all_paths)
        # print(self.all_paths)

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
                o_str = "epoch,loss,train-accuracy,test-accuracy"
            else:
                o_str = "epoch,loss,train-accuracy"

            outfile.write(o_str + "\n")

        for epoch in range(self.epochs):
            epoch_loss = 0
            n_correct = 0
            example_cnt = 0
            batch_cnt = 0
            self.cnts = [0 for i in range(len(self.layers_))]

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
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt
                    )
                    pbar.set_description(desc)
                
                print("")
                print("Leaf cnts are {} with total sum of {}".format(self.cnts[n_inner:], sum(self.cnts[n_inner:])))
                #print("Leaf cnts are {} with total sum of {}".format(self.cnts, sum(self.cnts)))
                if self.x_test is not None:
                    # output = apply_in_batches(self, self.x_test)
                    # accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    output = apply_in_batches(self, self.x_test, batch_size = self.batch_size)
                    accuracy_test_apply = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0

                    output_proba = self.predict_proba(self.x_test)
                    accuracy_test_proba = accuracy_score(np.argmax(output_proba, axis=1),self.y_test)*100.0

                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f} test acc {:2.4f} test acc proba {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        accuracy_test_apply,
                        accuracy_test_proba
                    )
                    pbar.set_description(desc)

            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                avg_loss = epoch_loss/example_cnt
                accuracy = 100.0*n_correct/example_cnt

                if self.x_test is not None:
                    output = apply_in_batches(self, self.x_test, batch_size = self.batch_size)
                    accuracy_test = accuracy_score(np.argmax(output, axis=1),self.y_test)*100.0
                    outfile.write("{},{},{},{}\n".format(epoch, avg_loss, accuracy, accuracy_test))
                else:
                    outfile.write("{},{},{}\n".format(epoch, avg_loss, accuracy))

    def forward(self, x):
        # Execute all models; This can be improved
        all_preds = [l(x) for l in self.layers_]
        path_preds = []
        for path in self.all_pathes:
            # print(path)
            pred = torch.tensor(1.0)
            for i in range(len(path[:-1])):
                cur_node = path[i]
                next_node = path[i+1]

                n_pred = all_preds[cur_node]
                tmp = n_pred.clone()
                tmp[tmp >= 0.5] = 1.0
                tmp[tmp < 0.5] = 0.0
                n_pred = tmp

                if cur_node == 0:
                    self.cnts[cur_node] += x.shape[0]
                else:
                    self.cnts[cur_node] += (pred != 0).sum().item()
                
                if (next_node % 2) == 0:
                    pred = n_pred * pred
                else:
                    pred = (1.0 - n_pred) * pred
            
            # print("LEAF IS: ",  path[-1])
            # if (path[-1] % 2) != 0:
            #     pred = 1.0 - pred

            self.cnts[path[-1]] += (pred != 0).sum().item()
            pred = pred * all_preds[path[-1]]

            # if (path[-1] % 2) == 0:
            #     print("EVEN")
            #     pred = pred * n_pred
            # else:
            #     print("UNEVEN")
            #     pred = pred * (1.0 - n_pred)
            # # Prediction of leaf node
            # print("node {} with {}".format(path[-1], (pred != 0).sum().item()))

            #self.cnts[path[-1]] += (pred != 0).sum().item()    
            # print(pred.shape)
            path_preds.append(pred)
        # asdf
        tmp = torch.stack(path_preds)
        # asdf
        # print(cur_path)
        # asdf
        # path = []
        # s = self.layers_[0](x)
        # # tmp = s.clone()
        # # tmp[tmp >= 0.5] = 1.0
        # # tmp[tmp < 0.5] = 0.0
        # # s = tmp
        # self.cnts[0] += x.shape[0]
        # self.cnts[1] += s.sum().item()
        # self.cnts[2] += (1.0-s).sum().item()
        # # self.cnts[1] += (s == 1.0).sum().item()
        # # self.cnts[2] += (s == 0.0).sum().item()
        # return s*self.layers_[1](x)+(1.0-s)*self.layers_[2](x)

        # path_preds = []
        # for path in self.all_paths:
        #     pred = torch.tensor(1.0)
        #     for was_left, node in path:
        #         # print("WAS LEFT: ", was_left)
        #         # print("EAVL", node["model"])
        #         m_pred = node["model"](x)
               
        #         if node["is_leaf"]:
        #             print(pred)
        #             node["cnt"] = node["cnt"] + (pred != 0).sum()
        #         else:
        #             # Since pytorch does not allow inplace operations we clone m_pred here
        #             tmp = m_pred.clone()
        #             tmp[tmp >= 0.5] = 1.0
        #             tmp[tmp < 0.5] = 0.0
        #             m_pred = tmp
        #             if node["invert"]:
        #                 m_pred = 1.0 - m_pred

        #         # print("BEFORE pred: ", pred)
        #         # print("m_pred:", m_pred)
        #         pred = m_pred*pred
        #         # print("AFTER pred:", pred)
        #         # asdf
        #             # l = m_pred >= 0.5
        #             # m_pred[l] = 1.0
        #             # m_pred[~l] = 0.0
        #             # print(l)
        #     # print(pred)
        #     path_preds.append(pred)
        # # asdf
        # 
        # # print(path_preds)
        # asdf
        
        return tmp.sum(dim = 0)
