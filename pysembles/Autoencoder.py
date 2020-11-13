from collections import OrderedDict
from functools import partial
import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
import torchvision.models as models

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score

from .Utils import Flatten, TransformTensorDataset, apply_in_batches, Scale
from .Models import BaseModel
from .BinarisedNeuralNetworks import binarize, BinaryTanh, BinaryLinear, BinaryConv2d

class UnpoolableMaxPool2d(nn.Module):
    def __init__(self, maxpool2d):
        super(UnpoolableMaxPool2d, self).__init__()
        self.maxpool2d = nn.MaxPool2d(
            kernel_size = maxpool2d.kernel_size,
            stride = maxpool2d.stride,
            padding = maxpool2d.padding,
            dilation = maxpool2d.dilation,
            ceil_mode = maxpool2d.ceil_mode,
            return_indices = True
        )

    def forward(self, x):
        self.output_size = x.shape
        x, self.pool_idx = self.maxpool2d(x)
        return x

class MaxUnpool2d(nn.Module):
    def __init__(self, unpoolable):
        super(MaxUnpool2d, self).__init__()
        self.unpoolable = unpoolable

    def forward(self, x):
        return nn.functional.max_unpool2d(
            x, 
            self.unpoolable.pool_idx, 
            kernel_size = self.unpoolable.maxpool2d.kernel_size,
            stride = self.unpoolable.maxpool2d.stride,
            padding = self.unpoolable.maxpool2d.padding,
            output_size = self.unpoolable.output_size
        )

class UnFlatten(nn.Module):
    def __init__(self, flatten):
        super(UnFlatten, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        return x.view(self.flatten.shape)

def encoder_decoder(encoder, decoder = None):
    # TODO THIS ASSUME THAT WE CAN ITERATE OVER THE MODEL. IS THIS CORRECT?
    enc = []
    for l in encoder():
        if isinstance(l, nn.MaxPool2d):
            enc.append(
                UnpoolableMaxPool2d(l)
            )
        elif isinstance(l, Flatten):
            enc.append(
                Flatten(store_shape=True)
            )
        else:
            enc.append(l)

    if decoder is None:
        dec = []

        for l in reversed(enc):
            # TODO: Add support for Binary Stuff
            # TODO: We should really copy everything without (named) arguments, shouldnt we?
            if isinstance(l, nn.Conv2d):
                dec.append(
                    nn.ConvTranspose2d(
                        l.out_channels, 
                        l.in_channels, 
                        kernel_size = l.kernel_size, 
                        stride = l.stride, 
                        padding = l.padding, 
                        dilation = l.dilation, 
                        groups = l.groups, 
                        bias = hasattr(l, 'bias'), 
                        padding_mode = l.padding_mode
                    )
                )
            elif isinstance(l, nn.Linear):
                dec.append(
                    nn.Linear(l.out_features, l.in_features, hasattr(l, 'bias'))
                )
            elif isinstance(l, UnpoolableMaxPool2d):
                dec.append(
                    MaxUnpool2d(l)
                )
            elif isinstance(l, Flatten):
                dec.append(
                    UnFlatten(l)
                )
            elif isinstance(l, Flatten):
                pass 
            else:
                dec.append(
                    l
                )
    else:
        dec = list(decoder())
        # for l in decoder():
        #     estimator.append(l)
    
    model = enc + dec
    return nn.Sequential(*model)

class Autoencoder(BaseModel):
    def __init__(self, encoder, decoder = None, *args, **kwargs):
        base_estimator = partial(encoder_decoder, encoder = encoder, decoder = decoder)
        super().__init__(base_estimator=base_estimator,*args, **kwargs)
        
        self.encoder = encoder
        
        tmp_enc = encoder()
        self.encoder_ = self.model[0:len(tmp_enc)]

    def restore_state(self,checkpoint):
        super().restore_state(checkpoint)
        
        tmp_enc = encoder()
        self.encoder_ = self.model[0:len(tmp_enc)]


    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "encoder":self.encoder
        } 

    def prepare_backward(self, data, target = None, weights = None):
        output = self(data)
        # print("Data shape is {}".format(data.shape))
        # print("Output shape is {}".format(output.shape))

        dim = 1.0
        for d in data.shape[1:]:
            dim *= d

        d = {
            "backward" : self.loss_function(data, output), 
            "metrics" :
            {
                "loss" : loss / dim
            } 
            
        }
        return d

    def forward(self, x):
        if self.training:
            # for i,l in enumerate(self.layers_):
            #     print("Running layer {} of type {} on shape {}".format(i, l.__class__.__name__, x.shape))
            #     x = l(x)
            # return x 
            return self.model(x)
        else:
            return self.encoder_(x)