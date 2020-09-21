# -*- coding: utf-8 -*-
"""
Copied  from https://github.com/Akashmathwani/Binarized-Neural-networks-using-pytorch/tree/master/MNIST%20using%20Binarized%20weights
"""

import math

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from torch.nn.modules.utils import _pair, _quadruple

class BinarizeF(Function):
    @staticmethod
    def forward(ctx, input):
        #input = input.clamp(-1,+1)
        #ctx.save_for_backward(input)
        output = input.new(input.size())
        output[input > 0] = 1
        output[input <= 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #return grad_output, None
        grad_input = grad_output.clone()
        return grad_input#, None

# aliases
binarize = BinarizeF.apply

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
