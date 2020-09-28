#!/usr/bin/env python3

import torch
import torch.optim as optim
from sklearn.base import clone
from torch import nn
from tqdm import tqdm

from .Models import SKEnsemble
from .Utils import TransformTensorDataset, apply_in_batches, cov, is_same_func

class GNCLClassifier(SKEnsemble):
    """ (Generalized) Negtaive Correlation Learning.

    Negative Correlation Learning uses the Bias-Variance-Co-Variance decomposition to derive a regularized objective function which enforces the diversity among the ensemble. 
    The first appearance of this work was for the MSE loss in Liu et al. in 1999 / Brown et al. in 2005. 
    Optiz later proposed a similar loss using the Cross Entropy Loss, but without theoretical justifications. Webb et al. in 2019/2020 gave more theoretical background to using the Cross Entropy Loss. Note that the 2020 Paper is basically a shorter version of the 2019 for the ECML PKDD conference. 

    We generalized the previous works to include _any_ loss function using a second order taylor approximation. However, this implementation currently supports three loss functions: MSE, Negative Log-Likelihood and CrossEntropy. 
    The main reasons for this are twofold: First, we manually have to compute the hessian for every loss function which is a lot of work. Second, PyTorchs Autograd functionality does not directly support a hessian (or even gradients) on a "per example" basis, but only summed over a batch. So far I have not found a way to use autograd efficiently here. 

    To make experiments more or less comparable, we opted for the formulation proposed by Brown et al. (which was later also used by Webb et al). They define the regularization strength l_reg (lambda):

    Loss = (1.0 - self.l_reg) * iloss - self.l_reg * 1.0/self.n_estimators * 0.5 * diversity_regularization
    
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
        - Liu, Y., & Yao, X. (1999). Ensemble learning via negative correlation. Neural Networks, 12(10), 1399–1404. https://doi.org/10.1016/S0893-6080(99)00073-8 
        - Brown, G., WatT, J. L., & Tino, P. (2005). Managing Diversity in Regression Ensembles. Jmlr, (6), 1621–1650. https://doi.org/10.1097/IYC.0000000000000008
        - Opitz, M., Possegger, H., & Bischof, H. (2016). Efficient model averaging for deep neural networks. Asian Conference on Computer Vision, 10112 LNCS, 205–220. https://doi.org/10.1007/978-3-319-54184-6_13
        - Shi, Z., Zhang, L., Liu, Y., Cao, X., Ye, Y., Cheng, M., & Zheng, G. (n.d.). Crowd Counting with Deep Negative Correlation Learning, 5382–5390. Retrieved from http://openaccess.thecvf.com/content_cvpr_2018/papers/Shi_Crowd_Counting_With_CVPR_2018_paper.pdf
        - Webb, A. M., Reynolds, C., Iliescu, D.-A., Reeve, H., Lujan, M., & Brown, G. (2019). Joint Training of Neural Network Ensembles, (4), 1–14. https://doi.org/10.13140/RG.2.2.28091.46880
        - Webb, A. M., Reynolds, C., Chen, W., Reeve, H., Iliescu, D.-A., Lujan, M., & Brown, G. (2020). To Ensemble or Not Ensemble: When does End-To-End Training Fail? In ECML PKDD 2020 (pp. 1–16). Retrieved from http://www.cs.man.ac.uk/~gbrown/publications/ecml2020webb.pdf
    """

    def __init__(self, l_reg = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l_reg = l_reg
        self.estimators_ = nn.ModuleList([ self.base_estimator() for _ in range(self.n_estimators)])

        assert self.l_reg >= 0 and self.l_reg <= 1, "Regularization parameter l_reg (lambda) must be between 0 and 1 vor GNCL Learning"

    def prepare_backward(self, data, target, weights = None):
        # TODO Make this use of the weights as well!
        f_bar, base_preds = self.forward_with_base(data)
        
        if isinstance(self.loss_function, nn.MSELoss): 
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]

            eye_matrix = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            D = 2.0*eye_matrix
        elif isinstance(self.loss_function, nn.NLLLoss):
            n_classes = f_bar.shape[1]
            n_preds = f_bar.shape[0]
            D = torch.eye(n_classes).repeat(n_preds, 1, 1).cuda()
            target_one_hot = torch.nn.functional.one_hot(target, num_classes = n_classes).type(torch.cuda.FloatTensor)

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
            # TODO Use autodiff do compute second derivative for given loss function
            # OR Use second formula from paper here? 
            D = torch.tensor(1.0)

        losses = []
        accuracies = []
        diversity = []
        for pred in base_preds:
            diff = pred - f_bar 
            covar = torch.bmm(diff.unsqueeze(1), torch.bmm(D, diff.unsqueeze(2))).squeeze()
            div = 1.0/self.n_estimators * 0.5 * covar
            i_loss = self.loss_function(pred, target)
            reg_loss = (1.0 - self.l_reg)*i_loss - self.l_reg * div
            
            losses.append(reg_loss)
            accuracies.append(100.0*(pred.argmax(1) == target).type(torch.cuda.FloatTensor))
            diversity.append(div)

        losses = torch.stack(losses, dim = 1)
        accuracies = torch.stack(accuracies, dim = 1)
        diversity = torch.stack(diversity, dim = 1)
        
        d = {
            "prediction" : f_bar, 
            "backward" : losses.sum(dim=1), 
            "metrics" :
            {
                "loss" : self.loss_function(f_bar, target),
                "accuracy" : 100.0*(f_bar.argmax(1) == target).type(torch.cuda.FloatTensor), 
                "avg loss": losses.mean(dim=1),
                "avg accuracy": accuracies.mean(dim = 1),
                "diversity": diversity.mean(dim = 1)
            } 
            
        }
        return d

