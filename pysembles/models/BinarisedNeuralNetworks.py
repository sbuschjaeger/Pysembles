# -*- coding: utf-8 -*-
"""
Copied  from https://github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch/tree/master/MNIST%20using%20Binarized%20weights
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair, _quadruple

import binarization

def binarize_to_plusminus1(input):
    input_shape = list(input.shape)
    len_input_shape = len(input_shape)

    if len_input_shape == 4:
        input = input.view(input_shape[0], input_shape[1], -1)
    if len_input_shape == 3:
        input = input.view(input_shape[0], input_shape[1], input_shape[2])
    if len_input_shape == 2:
        input = input.view(input_shape[0], input_shape[1], 1)
    if len_input_shape == 1:
        input = input.view(input_shape[0], 1, 1)

    inputL = binarization.binarization(input)
    input = inputL[0].view(input_shape)

    return input

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input, flip_prob = None):
        #input = input.clamp(-1,+1)
        #ctx.save_for_backward(input)
        output = input.clone()
        output = binarize_to_plusminus1(output)
        # if flip_prob is not None:
        # output = fi_binarized_float_plusminus1(output, flip_prob, flip_prob)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output, None
        grad_input = grad_output.clone()
        return grad_input, None

# aliases
binarize = BinarizeF.apply

# class BinarizeF(Function):
#     @staticmethod
#     def forward(ctx, input):
#         #input = input.clamp(-1,+1)
#         #ctx.save_for_backward(input)
#         output = input.new(input.size())
#         output[input > 0] = 1
#         output[input <= 0] = -1
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         #return grad_output, None
#         grad_input = grad_output.clone()
#         return grad_input#, None

# # aliases
# binarize = BinarizeF.apply

# def replace_modules(model, keep_activation, copy_init):
#     new_modules = []
#     for m in model.children():
#         children = [m for m in m.children()]

#         if len(children) > 1:
#             new_modules += replace_modules(m, keep_activation, copy_init)
#         elif not keep_activation and isinstance(m, (nn.ReLU, nn.Tanh)):
#             new_modules.append(BinaryTanh())
#         elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             new_modules.append(m)
#         elif isinstance(m, nn.Linear):
#             new_modules.append(BinaryLinear(m.in_features, m.out_features, hasattr(m, 'bias')))
#             if copy_init:
#                 if (hasattr(m, 'bias')):
#                     new_modules[-1].bias.data = m.bias.data
#                 new_modules[-1].weight.data = m.weight.data
#         elif isinstance(m, nn.Conv2d):
#             new_modules.append(BinaryConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, hasattr(m, 'bias'), m.padding_mode))
#             if copy_init:
#                 if (hasattr(m, 'bias')):
#                     new_modules[-1].bias.bias = m.bias.data 
#                 new_modules[-1].weight.data = m.weight.data
#         else:
#             new_modules.append(m)
#     return new_modules

def convert_layers(model, keep_activation, copy_init):
    for name, m in reversed(model._modules.items()):
        if len(list(m.children())) > 0:
            model._modules[name] = convert_layers(m, keep_activation, copy_init)

        if isinstance(m, nn.Linear):
            # layer_old = m
            layer_new = BinaryLinear(m.in_features, m.out_features, hasattr(m, 'bias'))
            if copy_init:
                if (hasattr(m, 'bias')):
                    layer_new.bias.data = m.bias.data
                layer_new.weight.data = m.weight.data
            model._modules[name] = layer_new

        if isinstance(m, nn.Conv2d):
            # layer_old = m
            layer_new = BinaryConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, hasattr(m, 'bias'), m.padding_mode)
            if copy_init:
                if (hasattr(m, 'bias')):
                    layer_new.bias.data = m.bias.data
                layer_new.weight.data = m.weight.data
            model._modules[name] = layer_new

        if not keep_activation and isinstance(m, (nn.ReLU, nn.Tanh)):
            # layer_old = m
            layer_new = BinaryTanh()
            model._modules[name] = layer_new

    return model

class BinaryModel(nn.Module):
    def __init__(self, model, keep_activation = False, include_scale = True, copy_init = False):
        super().__init__()

        self.model = convert_layers(model, keep_activation, copy_init)
        self.include_scale = include_scale
        
        # This has some rather crude assumptions. Lets see how they work out. 
        # First, we assume that the first module is either the network itself (e.g. VGGNet) or it is
        # an nn.Sequential. Second, we assume that modules() will provide a list of all layers and we do not
        # need to work on them recursivley?
        # modules = model.modules()
        # next(modules)

        # for m in model.children():
        #     print("CHECKING ", m)
        #     if isinstance(m, nn.Sequential):
        #         pass
        #     elif not keep_activation and isinstance(m, (nn.ReLU, nn.Tanh)):
        #         new_modules.append(BinaryTanh())
        #     elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        #         new_modules.append(m)
        #     elif isinstance(m, nn.Linear):
        #         print("REPLACING LINEAR")
        #         new_modules.append(BinaryLinear(m.in_features, m.out_features, hasattr(m, 'bias')))
        #         if copy_init:
        #             if (hasattr(m, 'bias')):
        #                 new_modules[-1].bias.data = m.bias.data
        #             new_modules[-1].weight.data = m.weight.data
        #     elif isinstance(m, nn.Conv2d):
        #         print("REPLACING Conv2D")
        #         new_modules.append(BinaryConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, hasattr(m, 'bias'), m.padding_mode))
        #         if copy_init:
        #             if (hasattr(m, 'bias')):
        #                 new_modules[-1].bias.bias = m.bias.data 
        #             new_modules[-1].weight.data = m.weight.data
        #     else:
        #         new_modules.append(m)
        if self.include_scale:
            self.scale = Scale()
    
    def forward(self, x):
        x = self.model(x)
        
        if self.include_scale:
            return self.scale(x)
        else:
            return x

class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class BinaryTanh(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh(*args, **kwargs)

    def forward(self, input):
        output = self.hardtanh(input)
        output = binarize(output)
        return output
        
class BinaryLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinaryLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)

            return F.linear(input, binary_weight)
        else:
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)
            return F.linear(input, binary_weight, binary_bias)

    def reset_parameters(self):
        #self.weight.data.uniform_(-5,+5)

        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv

class BinaryConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(BinaryConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            binary_weight = binarize(self.weight)
            return F.conv2d(input, binary_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            binary_weight = binarize(self.weight)
            binary_bias = binarize(self.bias)
            
            return F.conv2d(input, binary_weight, binary_bias, self.stride,
                            self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv
