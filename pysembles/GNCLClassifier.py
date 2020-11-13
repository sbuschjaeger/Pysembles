#!/usr/bin/env python3

import warnings
import torch
import torch.optim as optim
from sklearn.base import clone
from torch import nn
from tqdm import tqdm

from .Models import Ensemble
from .Utils import TransformTensorDataset

class GNCLClassifier(Ensemble):
    """ (Generalized) Negtaive Correlation Learning.

    Negative Correlation Learning uses the Bias-Variance-Co-Variance decomposition to derive a regularized objective function which enforces the diversity among the ensemble. 
    The first appearance of this work was for the MSE loss in Liu et al. in 1999 / Brown et al. in 2005. 
    Optiz later proposed a similar loss using the Cross Entropy Loss, but without theoretical justifications. Webb et al. in 2019/2020 gave more theoretical background to using the Cross Entropy Loss. Note that the 2020 Paper is basically a shorter version of the 2019 for the ECML PKDD conference. 

    We generalized the previous works to include _any_ loss function using a second order taylor approximation. However, this implementation currently supports three loss functions: MSE, Negative Log-Likelihood and CrossEntropy. 
    The main reasons for this are twofold: First, we manually have to compute the hessian for every loss function which is a lot of work. Second, PyTorchs Autograd functionality does not directly support a hessian (or even gradients) on a "per example" basis, but only summed over a batch. So far I have not found a way to use autograd efficiently here. 

    To make experiments more or less comparable, we opted for the formulation proposed by Brown et al. They define the regularization strength l_reg (which is kappa in eq (17) in [2]):

    Loss = \sum_{i=1}^self.n_estimators 1.0 / self.n_estimators * iloss - self.l_reg * 1.0/self.n_estimators * 0.5 * diversity_regularization
    
    where 
        - Loss: The loss which is minimized
        - diversity_regularization: Expresses the diversity of the ensemble wrt. to the currently loss function (MSE, CrossEntropy, NLL etc.)
        - iloss: The invidual loss (MSE, CrossEntropy, NLL etc.) of each ensemble member 
        - self.l_reg: The trade-off between diversity and loss.
        - self.n_estimators: Number of ensemble members
        
    Attributes:
        n_estimators (int): Number of estimators in ensemble. Should be at least 1
        l_reg (float): Trade-off between diversity and loss. Should be between 0 and 1. l_reg = 0 implies independent training whereas l_reg = 1 implies no independent training

    References
        [1] Liu, Y., & Yao, X. (1999). Ensemble learning via negative correlation. Neural Networks, 12(10), 1399–1404. https://doi.org/10.1016/S0893-6080(99)00073-8 
        [2] Brown, G., WatT, J. L., & Tino, P. (2005). Managing Diversity in Regression Ensembles. Jmlr, (6), 1621–1650. https://doi.org/10.1097/IYC.0000000000000008
        [3] Opitz, M., Possegger, H., & Bischof, H. (2016). Efficient model averaging for deep neural networks. Asian Conference on Computer Vision, 10112 LNCS, 205–220. https://doi.org/10.1007/978-3-319-54184-6_13
        [4] Shi, Z., Zhang, L., Liu, Y., Cao, X., Ye, Y., Cheng, M., & Zheng, G. (n.d.). Crowd Counting with Deep Negative Correlation Learning, 5382–5390. Retrieved from http://openaccess.thecvf.com/content_cvpr_2018/papers/Shi_Crowd_Counting_With_CVPR_2018_paper.pdf
        [5] Webb, A. M., Reynolds, C., Iliescu, D.-A., Reeve, H., Lujan, M., & Brown, G. (2019). Joint Training of Neural Network Ensembles, (4), 1–14. https://doi.org/10.13140/RG.2.2.28091.46880
        [6] Webb, A. M., Reynolds, C., Chen, W., Reeve, H., Iliescu, D.-A., Lujan, M., & Brown, G. (2020). To Ensemble or Not Ensemble: When does End-To-End Training Fail? In ECML PKDD 2020 (pp. 1–16). Retrieved from http://www.cs.man.ac.uk/~gbrown/publications/ecml2020webb.pdf
    """

    def __init__(self, l_reg = 0, mode = "exact", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l_reg = l_reg
        self.mode = mode
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])

        if self.mode == "exact" and not isinstance(self.loss_function, (nn.MSELoss, nn.NLLLoss, nn.CrossEntropyLoss)):
            warnings.warn("You set GNCL to 'exact' but used an unsupported loss function for exact minimization. Currrently supported are MSELoss, NLLLoss, and CrossEntropyLoss. I am setting mode to 'upper' now and minimize the upper bound using the provided loss function") 
            self.mode = "upper"

    def restore_state(self, checkpoint):
        super().restore_state(checkpoint)
        self.mode = checkpoint["mode"]

    def get_state(self):
        state = super().get_state()
        return {
            **state,
            "mode":self.mode,
        } 

    def prepare_backward(self, data, target, weights = None):
        # TODO Make this use of the weights as well!
        f_bar, base_preds = self.forward_with_base(data)
        
        if self.mode == "upper":
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]
            D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
        else:
            if isinstance(self.loss_function, nn.MSELoss): 
                n_classes = f_bar.shape[1]
                n_preds = f_bar.shape[0]

                eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
                D = 2.0*eye_matrix
            elif isinstance(self.loss_function, nn.NLLLoss):
                n_classes = f_bar.shape[1]
                n_preds = f_bar.shape[0]
                D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
                target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes).type(self.get_float_type())

                eps = 1e-7
                diag_vector = target_one_hot*(1.0/(f_bar**2+eps))
                D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
            elif isinstance(self.loss_function, nn.CrossEntropyLoss):
                n_preds = f_bar.shape[0]
                n_classes = f_bar.shape[1]
                f_bar_softmax = nn.functional.softmax(f_bar,dim=1)
                D = -1.0*torch.bmm(f_bar_softmax.unsqueeze(2), f_bar_softmax.unsqueeze(1))
                diag_vector = f_bar_softmax*(1.0-f_bar_softmax)
                D.diagonal(dim1=-2, dim2=-1).copy_(diag_vector)
            else:
                # NOTE: We should never reach this code path
                raise ValueError("Invalid combination of mode and loss function in GNCLClassifier.")

        losses = []
        accuracies = []
        diversity = []
        f_loss = self.loss_function(f_bar, target)
        for pred in base_preds:
            diff = pred - f_bar 
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/self.n_estimators * 1.0/2.0 * covar
            i_loss = self.loss_function(pred, target)

            if self.mode == "exact":
                # Eq. (4)
                reg_loss = 1.0/self.n_estimators * i_loss - self.l_reg * div
            else:
                # Eq. (5) where we scale the ensemble loss with 1.0/self.n_estimators due to the summation in line 118
                reg_loss = 1.0/self.n_estimators * self.l_reg * f_loss + (1.0 - self.l_reg)/self.n_estimators * i_loss
            
            losses.append(reg_loss)
            accuracies.append(100.0*(pred.argmax(1) == target).type(self.get_float_type()))
            diversity.append(div)

        losses = torch.stack(losses, dim = 1)
        accuracies = torch.stack(accuracies, dim = 1)
        diversity = torch.stack(diversity, dim = 1)
        
        # NOTE: avg loss is the average (regularized) loss and not the average loss (wrt. to the loss_function)
        d = {
            "prediction" : f_bar, 
            "backward" : losses.sum(dim=1), 
            "metrics" :
            {
                "loss" : f_loss,
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(self.get_float_type()), 
                "avg loss": losses.mean(dim=1),
                "avg accuracy": accuracies.mean(dim = 1),
                "diversity": diversity.sum(dim = 1)
            } 
            
        }
        return d

