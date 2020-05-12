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
from .Models import SKLearnModel
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

class Autoencoder(SKLearnModel):
    def __init__(self, encoder, decoder = None, *args, **kwargs):
        base_estimator = partial(encoder_decoder, encoder = encoder, decoder = decoder)
        super().__init__(base_estimator=base_estimator,*args, **kwargs)
        
        tmp_enc = encoder()
        self.encoder = self.layers_[0:len(tmp_enc)]

    def fit(self, X, y = None, sample_weight = None):
        # if y is not None:
        #     raise ValueError("The autoencoder should not receive any labels!") 

        x_tensor = torch.tensor(X)

        if sample_weight is not None:
            sample_weight = len(y)*sample_weight/np.sum(sample_weight)
            w_tensor = torch.tensor(sample_weight)
            w_tensor = w_tensor.type(torch.FloatTensor)
            data = TransformTensorDataset(x_tensor, w_tensor, transform=self.transformer)
        else:
            w_tensor = None
            data = TransformTensorDataset(x_tensor, transform=self.transformer)

        self.X_ = X

        # We have to set this variable to make the auto-encoder SKLEarn compatible
        self.y_ = None

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
            o_str = "epoch,train-loss"

            outfile.write(o_str + "\n")

        for epoch in range(self.epochs):
            epoch_loss = 0
            example_cnt = 0
            batch_cnt = 0

            with tqdm.tqdm(total=len(train_loader.dataset), ncols=135, disable = not self.verbose) as pbar:
                for batch in train_loader:
                    if sample_weight is not None:
                        data = batch[0]
                        weights = batch[1]

                        weights = weights.cuda()
                        weights = Variable(weights)
                    else:
                        data = batch

                    data = data.cuda()
                    data = Variable(data)

                    optimizer.zero_grad()
                    output = self(data)
                    # print("Data shape is {}".format(data.shape))
                    # print("Output shape is {}".format(output.shape))

                    dim = 1.0
                    for d in data.shape[1:]:
                        dim *= d

                    if sample_weight is not None: 
                        loss = self.loss_function(data, output, weights)
                        epoch_loss += loss.sum().item() / dim
                        loss = loss.mean()
                    else:
                        loss = self.loss_function(data, output)
                        epoch_loss += loss.sum().item() / dim
                        loss = loss.mean()
                    
                    loss.backward()
                    optimizer.step()

                    pbar.update(data.shape[0])
                    example_cnt += data.shape[0]
                    batch_cnt += 1
                    desc = '[{}/{}] loss {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt
                    )
                    pbar.set_description(desc)

            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                outfile.write("{},{}\n".format(epoch, epoch_loss/example_cnt))
    
    # For convenience we override predict and use predict_proba. 
    def predict(self, X, eval_mode=True):
        return self.predict_proba(X,eval_mode)

    def forward(self, x):
        if self.training:
            # for i,l in enumerate(self.layers_):
            #     print("Running layer {} of type {} on shape {}".format(i, l.__class__.__name__, x.shape))
            #     x = l(x)
            # return x 
            return self.layers_(x)
        else:
            return self.encoder(x)