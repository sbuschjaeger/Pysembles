# from collections import OrderedDict
from functools import partial

import numpy as np
import copy 

import torch
from torch import nn

"""
Boosting requires the weighting of each sample in the loss functions. This is not directly supported by PyTorch. 
Thus, for some common ones we provide implementations here. 
If you want to implement / add your own there are two important things to keep in mind:
    (1) Do not aggregate the loss over the batch size, but return a tensor with shape (batch_size, 1). 
        E.g. set reduction="none" for PyTorch losses
    (2) Weights can be None which means there are no weights (e.g. when Boosting is not used)
"""

def weighted_exp_loss(prediction, target, weights = None):
    prediction = prediction.type(torch.cuda.FloatTensor)
    num_classes = prediction.shape[1]
    target_one_hot = 2*torch.nn.functional.one_hot(target, num_classes = num_classes).type(torch.cuda.FloatTensor) - 1.0
    inner = target_one_hot*prediction
    return  torch.exp(-inner)

def weighted_squared_hinge_loss(prediction, target, weights = None):
    #torch.autograd.set_detect_anomaly(True)
    prediction = prediction.type(torch.cuda.FloatTensor)
    num_classes = prediction.shape[1]
    target_one_hot = 2*torch.nn.functional.one_hot(target, num_classes = num_classes).type(torch.cuda.FloatTensor) - 1.0
    inner = target_one_hot*prediction
    # Copy to prevent modified inplace error
    tmp = torch.zeros_like(inner)
    c1 = inner <= 0
    # "simulate" the "and" operations with "*" here
    c2 = (0 < inner) * (inner < 1)
    #c3 = inner >= 1
    tmp[c1] = 0.5-inner[c1]
    tmp[c2] = 0.5*(1-inner[c2])**2
    tmp = tmp.sum(axis=1)
    #tmp[c3] = 0

    if weights is None:
        return tmp
    else:
        return weights*tmp

def weighted_mse_loss(prediction, target, weights = None):
    criterion = nn.MSELoss(reduction="none")
    num_classes = prediction.shape[1]
    target_one_hot = torch.nn.functional.one_hot(target, num_classes = num_classes).type(torch.cuda.FloatTensor)

    unweighted_loss = criterion(prediction, target_one_hot)
    if weights is None:
        return unweighted_loss
    else:
        return weights*unweighted_loss

def weighted_cross_entropy(prediction, target, weights = None):
    num_classes = prediction.shape[1]
    target_one_hot = torch.nn.functional.one_hot(target, num_classes = num_classes).type(torch.cuda.FloatTensor)
    eps = 1e-7
    unweighted_loss = -(torch.log(prediction+eps)*target_one_hot).sum(dim=1)

    if weights is None:
        return unweighted_loss
    else:
        return weights*unweighted_loss

def weighted_cross_entropy_with_softmax(prediction, target, weights = None):
    criterion = nn.CrossEntropyLoss(reduction="none")
    unweighted_loss = criterion(prediction, target)
    if weights is None:
        return unweighted_loss
    else:
        return weights*unweighted_loss

def weighted_lukas_loss(prediction, target, weights = None):
    #1/(E^x^2 Sqrt[Pi]) + x + x Erf[x]
    #2/(E^x^2 Sqrt[Pi])
    num_classes = prediction.shape[1]
    target_one_hot = 2*torch.nn.functional.one_hot(target, num_classes = num_classes).type(torch.cuda.FloatTensor) - 1.0

    z = -prediction * target_one_hot
    return torch.sum(torch.exp(-z**2) *1.0/np.sqrt(np.pi) + z * (1 + torch.erf(z)),dim=1)