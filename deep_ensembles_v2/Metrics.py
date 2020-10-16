import numpy as np
import pandas as pd
import torch
import scipy

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import make_scorer, accuracy_score

from deep_ensembles_v2.Utils import TransformTensorDataset

def diversity(model, x, y):
    # This is basically a copy/paste from the GNCLClasifier regularizer, which can also be used for 
    # other classifier. I tried to do it with numpy first and I think it should work but I did not 
    # really understand numpy's bmm variant, so I opted for the safe route here. 
    # Also, pytorch seems a little faster due to gpu support
    if not hasattr(model, "estimators_"):
        return 0
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    diversities = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            f_bar, base_preds = model.forward_with_base(data)
        
        if isinstance(model.loss_function, nn.MSELoss): 
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]

            eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            D = 2.0*eye_matrix
        elif isinstance(model.loss_function, nn.NLLLoss):
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]
            D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes).type(torch.cuda.FloatTensor)

            eps = 1e-7
            diag_vector = target_one_hot*(1.0/(f_bar**2+eps))
            D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
        elif isinstance(model.loss_function, nn.CrossEntropyLoss):
            n_preds = f_bar.shape[0]
            n_classes = f_bar.shape[1]
            f_bar_softmax = nn.functional.softmax(f_bar,dim=1)
            D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
            diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
            D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
        else:
            D = torch.tensor(1.0)

        batch_diversities = []
        for pred in base_preds:
            diff = pred - f_bar 
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/model.n_estimators * 0.5 * covar
            batch_diversities.append(div)

        diversities.append(torch.stack(batch_diversities, dim = 1))
    div = torch.cat(diversities,dim=0)
    return div.sum(dim=1).mean(dim=0).item()
    
    # dsum = torch.sum(torch.cat(diversities,dim=0), dim = 0)
    # return dsum
    # base_preds = []
    # for e in model.estimators_:
    #     ypred = apply_in_batches(e, x, 128)
    #     base_preds.append(ypred)
    
    # f_bar = np.mean(base_preds, axis=0)
    # if isinstance(model.loss_function, nn.MSELoss): 
    #     n_classes = f_bar.shape[1]
    #     n_preds = f_bar.shape[0]

    #     eye_matrix = np.eye(n_classes).repeat(n_preds, 1, 1)
    #     D = 2.0*eye_matrix
    # elif isinstance(model.loss_function, nn.NLLLoss):
    #     n_classes = f_bar.shape[1]
    #     n_preds = f_bar.shape[0]
    #     D = np.eye(n_classes).repeat(n_preds, 1, 1)
    #     target_one_hot = np.zeros((y.size, n_classes))
    #     target_one_hot[np.arange(y.size),y] = 1

    #     eps = 1e-7
    #     diag_vector = target_one_hot*(1.0/(f_bar**2+eps))
    #     #D[np.diag_indices(D.shape[0])] = diag_vector
    #     for i in range(D.shape[0]):
    #         np.fill_diagonal(D[i,:], diag_vector[i,:])
    # elif isinstance(model.loss_function, nn.CrossEntropyLoss):
    #     n_preds = f_bar.shape[0]
    #     n_classes = f_bar.shape[1]
    #     f_bar_softmax = scipy.special.softmax(f_bar,axis=1)

    #     D = -1.0 * np.expand_dims(f_bar_softmax, axis=2) @ np.expand_dims(f_bar_softmax, axis=1)

    #     # D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
    #     diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
    #     for i in range(D.shape[0]):
    #         np.fill_diagonal(D[i,:], diag_vector[i,:])
    # else:
    #     D = np.array([1.0])

    # diversities = []
    # for pred in base_preds:
    #     # https://stackoverflow.com/questions/63301019/dot-product-of-two-numpy-arrays-with-3d-vectors
    #     # https://stackoverflow.com/questions/51479148/how-to-perform-a-stacked-element-wise-matrix-vector-multiplication-in-numpy
    #     diff = pred - f_bar 
    #     tmp = np.sum(D * diff[:,:,None], axis=1)
    #     covar = np.sum(tmp*diff,axis=1)

    #     # covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
    #     div = 1.0/model.n_estimators * 0.5 * covar
    #     diversities.append(np.mean(div))
    #return np.sum(diversities)

def loss(model, x, y):
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    losses = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            pred = model(data)
        
        losses.append(model.loss_function(pred, target).mean().item())
    
    return np.mean(losses)

def avg_loss(model, x, y):
    if not hasattr(model, "estimators_"):
        return 0
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    losses = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            f_bar, base_preds = model.forward_with_base(data)
        
        ilosses = []
        for base in base_preds:
            ilosses.append(model.loss_function(base, target).mean().item())
            
        losses.append(np.mean(ilosses))

    return np.mean(losses)

def avg_accurcay(model, x, y):
    if not hasattr(model, "estimators_"):
        return 0
    model.eval()
    
    x_tensor = torch.tensor(x)
    y_tensor = torch.tensor(y)
    dataset = TransformTensorDataset(x_tensor, y_tensor, transform=None)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size = model.batch_size)
    
    accuracies = []
    for batch in test_loader:
        data, target = batch[0], batch[1]
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            _, base_preds = model.forward_with_base(data)
        
        iaccuracies = []
        for base in base_preds:
            iaccuracies.append( 100.0*(base.argmax(1) == target).type(torch.cuda.FloatTensor) )
            
        accuracies.append(torch.cat(iaccuracies,dim=0).mean().item())

    return np.mean(accuracies)
    # accuracies = torch.cat(accuracies,dim=0)
    # return accuracies.mean().item()