from collections import OrderedDict
import numpy as np
import torch
import tqdm
from torch import nn
from torch.autograd import Variable
import torchvision.models as models

from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from Utils import Flatten, TransformTensorDataset, apply_in_batches
from Models import SKLearnModel

from BinarisedNeuralNetworks import binarize, BinaryTanh, BinaryLinear, BinaryConv2d

class SwitcherooModel(SKLearnModel):
    def __init__(self, switch_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch_epoch = switch_epoch

    def fit(self, X, y, sample_weight = None):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
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
                o_str = "epoch,train-loss,train-accuracy,test-loss,test-accuracy"
            else:
                o_str = "epoch,train-loss,train-accuracy"

            outfile.write(o_str + "\n")

        for epoch in range(self.epochs):
            # Perform the switcheroo
            if epoch == self.switch_epoch:
                new_modules = OrderedDict()
                for m_name, m in self._modules.items():
                    if isinstance(m, nn.ReLU):
                        new_modules[m_name] = BinaryTanh()
                    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        new_modules[m_name] = m
                    elif isinstance(m, nn.Linear):
                        new_modules[m_name] = BinaryLinear(m.in_features, m.out_features, hasattr(m, 'bias'))
                        
                        if (hasattr(m, 'bias')):
                            new_modules[m_name].bias.data = binarize(m.bias).data
                        new_modules[m_name].weight.data = binarize(m.weight).data
                    elif isinstance(m, nn.Conv2d):
                        new_modules[m_name] = BinaryConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, hasattr(m, 'bias'), m.padding_mode)
                        if (hasattr(m, 'bias')):
                            new_modules[m_name].bias.bias = binarize(m.bias).data
                        new_modules[m_name].weight.data = binarize(m.weight).data
                    else:
                        new_modules[m_name] = m
                
                self.layers_ = nn.Sequential(new_modules)
                self.cuda()
                self.train()
                optimizer = self.optimizer_method(self.parameters(), **self.optimizer)

                # TODO THIS RESETS THE SCHEDULER!
                if self.scheduler_method is not None:
                    scheduler = self.scheduler_method(optimizer, **self.scheduler)
                else:
                    scheduler = None
                    
            epoch_loss = 0
            n_correct = 0
            example_cnt = 0
            batch_cnt = 0
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
                    # print("")
                    # print(self.layers_[-3].running_var)
                    # print(self.layers_[-3].running_mean)
                    # print("")
                    desc = '[{}/{}] loss {:2.4f} acc {:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt
                    )
                    pbar.set_description(desc)
            
                if self.x_test is not None and epoch % self.eval_test == 0:
                    pred_proba = self.predict_proba(self.x_test)
                    pred_tensor = torch.tensor(pred_proba).cuda()
                    y_test_tensor = torch.tensor(self.y_test).cuda()
                    test_loss = self.loss_function(pred_tensor, y_test_tensor).mean().item()
                    accuracy_test = accuracy_score(self.y_test, np.argmax(pred_proba, axis=1))*100.0  
                    
                    desc = '[{}/{}] loss {:2.4f} train acc {:2.4f} test loss/acc {:2.4f}/{:2.4f}'.format(
                        epoch, 
                        self.epochs-1, 
                        epoch_loss/example_cnt, 
                        100. * n_correct/example_cnt,
                        test_loss,
                        accuracy_test
                    )
                    pbar.set_description(desc)

            if scheduler is not None:
                scheduler.step()

            torch.cuda.empty_cache()
            
            if self.out_path is not None:
                avg_loss = epoch_loss/example_cnt
                accuracy = 100.0*n_correct/example_cnt

                if self.x_test is not None:
                    if epoch % self.eval_test != 0:
                        accuracy_test = "-"
                        test_loss = "-"

                    outfile.write("{},{},{},{},{}\n".format(epoch, avg_loss, accuracy, test_loss, accuracy_test))
                else:
                    outfile.write("{},{},{}\n".format(epoch, avg_loss, accuracy))
        
    def forward(self, x):
        return self.layers_(x)