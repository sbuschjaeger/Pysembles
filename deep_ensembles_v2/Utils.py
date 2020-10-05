# from collections import OrderedDict
from functools import partial
import inspect
import warnings

import numpy as np
import copy 

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.autograd import Variable

# from torch.autograd import Variable
# from torch.optim.optimizer import Optimizer, required

import torchvision
import torchvision.transforms as transforms

from .BinarisedNeuralNetworks import binarize, BinaryTanh, BinaryLinear, BinaryConv2d

# This function can be used as a scoring metric to score the number of parameters
def pytorch_total_params(model, x, y):
    return sum(p.numel() for p in model.parameters())

def flatten_dict(d):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_d = flatten_dict(v)
            for k2, v2 in flat_d.items():
                flat_dict[k + "_" + k2] = v2
        else:
            flat_dict[k] = v
    return flat_dict

def replace_objects(d):
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = replace_objects(v)
        elif isinstance(v, partial):
            d[k] = v.func.__name__ + "_" + "_".join([str(arg) for arg in v.args])
        elif callable(v) or inspect.isclass(v):
            try:
                d[k] = v.__name__
            except Exception as e:
                d[k] = str(v) #.__name__
    return d

def dict_to_str(d):
    return str(replace_objects(d)).replace(":","=").replace(",","_").replace("\"","").replace("\'","").replace("{","").replace("}","").replace(" ", "")

def replace_layer_if_possible(layer):
    class Sign(nn.Module):
        def __init__(self):
            super(Sign, self).__init__()

        def forward(self, input):
            return torch.where(input > 0, torch.tensor([1.0]).cuda(), torch.tensor([-1.0]).cuda())
    
    print("FOUND ", layer)
    if isinstance(layer, BinaryTanh):
        print("REPLACING BINARY TANH")
        new_layer = Sign()
    elif isinstance(layer, BinaryLinear):
        print("REPLACING LIN LAYER")
        new_layer = nn.Linear(layer.in_features, layer.out_features, hasattr(layer, 'bias'))
        if hasattr(layer, 'bias'):
            new_layer.bias.data = binarize(layer.bias).data
        new_layer.weight.data = binarize(layer.weight).data
    elif isinstance(layer, BinaryConv2d):
        print("REPLACING CONV LAYER")
        new_layer = nn.Conv2d(
            layer.in_channels, layer.out_channels, layer.kernel_size, 
            layer.stride, layer.padding, layer.dilation, layer.groups, 
            hasattr(layer, 'bias'), layer.padding_mode
        )
        if hasattr(layer, 'bias'):
            new_layer.bias.data = binarize(layer.bias).data
        new_layer.weight.data = binarize(layer.weight).data
    else:
        new_layer = layer
    return new_layer

def replace_sequential_if_possible(s):
    for i,si in enumerate(s):
        print("CHECKING ", si)
        if hasattr(s[i], "layers_"):
            print("LAYERS_ FOUND, REPLACING")
            s[i].layers_ = replace_sequential_if_possible(s[i].layers_)
        if isinstance(s[i], nn.Sequential):
            print("SEQUENTIAL FOUND; REPLACING")
            s[i] = replace_sequential_if_possible(s[i])
        else:
            print("REGULAR LAYER FOUND")
            s[i] = replace_layer_if_possible(s[i])
        # new_seq.append(tmp)
    return s

def store_model(model, path, dim, verbose = False):
    # Since we change layers in-place we copy it beforehand
    model = copy.deepcopy(model)
    print("BEFORE REPLACE:", model)

    model.layers_ = replace_sequential_if_possible(model.layers_)
    # for i in range(len(model.layers_)):
    #     model.layers_[i] = replace_layer_if_possible(model.layers_[i])

    #new_modules = OrderedDict()
    #for m_name, m in model._modules.items():
    # for m in model.modules():
    #     #new_modules[m_name] = replace_if_possible(m)
    #     if isinstance(m, nn.Sequential):
    #         m = replace_sequential_if_possible(m)
    #     else:
    #         m = replace_layer_if_possible(m)
    #     #m = replace_if_possible(m)
    
    print("AFTER REPLACE:", model)
    dummy_input = torch.randn((1,*dim), device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model.cuda(), dummy_input, path, verbose=verbose, input_names=["input"], output_names=["output"])

# See: https://github.com/pytorch/pytorch/issues/19037
def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact
    return c.squeeze()

def is_same_func(f1,f2):
    if isinstance(f1, partial) and isinstance(f2, partial):
        return f1.func == f2.func and f1.args == f2.args and f1.keywords == f2.keywords
    elif isinstance(f1, partial):
        return f1.func == f2
    elif isinstance(f1, partial):
        return f2.func == f1
    else:
        return f1 == f2

def apply_in_batches(fun, X, batch_size):
    x_tensor = torch.tensor(X)

    dataset = TransformTensorDataset(x_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    
    values = None
    for batch in test_loader:
        test_data = batch
        test_data = test_data.cuda()
        test_data = Variable(test_data)

        val = fun(test_data)
        val = val.cpu().detach().numpy()
        
        if values is None:
            values = val
        else:
            values = np.concatenate( (values, val), axis=0 )

    return values

# See: https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset/55593757
class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, x_tensor, y_tensor = None, w_tensor = None, transform=None):
        self.x = x_tensor
        self.y = y_tensor
        self.w = w_tensor
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform is not None:
            x = self.transform(x)
        
        if self.w is not None:
            y = self.y[index]
            w = self.w[index]
            return x, y, w
        elif self.y is not None:
            y = self.y[index]

            return x, y
        else:
            return x

    def __len__(self):
        return self.x.size(0)

class Clippy(torch.optim.Adam):
    def step(self, closure=None):
        loss = super(Clippy, self).step(closure=closure)
        for group in self.param_groups:
            for p in group['params']:
                p.data.clamp(-1,1)
            
        return loss

# SOME HELPFUL LAYERS
class Flatten(nn.Module):
    def __init__(self, store_shape=False):
        super(Flatten, self).__init__()
        self.store_shape = store_shape

    def forward(self, x):
        if self.store_shape:
            self.shape = x.shape

        return x.flatten(1)
        #return x.view(x.size(0), -1)

class Clamp(nn.Module):
    def __init__(self, min_out = -3, max_out = 3):
        super().__init__()
        self.min_out = min_out
        self.max_out = max_out

    def forward(self, input):
        return input.clamp(self.min_out, self.max_out)

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class SkipConnection(nn.Module):
    def __init__(self, *block):
        super().__init__()
        self.layers_ = torch.nn.Sequential(*block)
    
    def forward(self, x):
        y = self.layers_(x)
        assert x.shape == y.shape
        return x + y