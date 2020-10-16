# PyTorch Ensembles 

This repository contains code for training ensembles with PyTorch. These are mostly straigt-forward implementations 
without much optimizations. I tried to stick as close as possible to the original papers to provide some references.
Currently supported are:

### Generalized Negative Correlation Learning
Negative Correlation Learning uses the Bias-Variance-Co-Variance decomposition to derive a regularized objective function which enforces the diversity among the ensemble. The first appearance of this work was for the MSE loss in Liu et al. in 1999 / Brown et al. in 2005. 
Optiz later proposed a similar loss using the Cross Entropy Loss, but without theoretical justifications. Webb et al. in 2019/2020 gave more theoretical background to using the Cross Entropy Loss. Note that the 2020 Paper is basically a shorter version of the 2019 for the ECML PKDD conference. 
We generalized the previous works to include _any_ loss function using a second order taylor approximation. However, this implementation currently supports three loss functions: MSE, Negative Log-Likelihood and CrossEntropy. This is work-in-progress!

References

- Liu, Y., & Yao, X. (1999). Ensemble learning via negative correlation. Neural Networks, 12(10), 1399–1404. https://doi.org/10.1016/S0893-6080(99)00073-8 
- Brown, G., WatT, J. L., & Tino, P. (2005). Managing Diversity in Regression Ensembles. Jmlr, (6), 1621–1650. https://doi.org/10.1097/IYC.0000000000000008
- Opitz, M., Possegger, H., & Bischof, H. (2016). Efficient model averaging for deep neural networks. Asian Conference on Computer Vision, 10112 LNCS, 205–220. https://doi.org/10.1007/978-3-319-54184-6_13
- Shi, Z., Zhang, L., Liu, Y., Cao, X., Ye, Y., Cheng, M., & Zheng, G. (n.d.). Crowd Counting with Deep Negative Correlation Learning, 5382–5390. Retrieved from http://openaccess.thecvf.com/content_cvpr_2018/papers/Shi_Crowd_Counting_With_CVPR_2018_paper.pdf
- Webb, A. M., Reynolds, C., Iliescu, D.-A., Reeve, H., Lujan, M., & Brown, G. (2019). Joint Training of Neural Network Ensembles, (4), 1–14. https://doi.org/10.13140/RG.2.2.28091.46880
- Webb, A. M., Reynolds, C., Chen, W., Reeve, H., Iliescu, D.-A., Lujan, M., & Brown, G. (2020). To Ensemble or Not Ensemble: When does End-To-End Training Fail? In ECML PKDD 2020 (pp. 1–16). Retrieved from http://www.cs.man.ac.uk/~gbrown/publications/ecml2020webb.pdf

### End2EndLearning
Directly E2E training of the entire ensemble. Just get the prediction of each base model, aggregate it and perform SGD on it as if it would be a new fancy Deep Learning architecture. Surprisingly, this approach is often overlooked in literature and sometimes it has strange names. I'll try to gather some references below, but apart from that there is nothing out of the ordinary to explain about this model compared to regular Deep architectures. 

References

- Lee, S., Purushwalkam, S., Cogswell, M., Crandall, D., & Batra, D. (2015). Why M Heads are Better than One: Training a Diverse Ensemble of Deep Networks. Retrieved from http://arxiv.org/abs/1511.06314
- Webb, A. M., Reynolds, C., Iliescu, D.-A., Reeve, H., Lujan, M., & Brown, G. (2019). Joint Training of Neural Network Ensembles, (4), 1–14. https://doi.org/10.13140/RG.2.2.28091.46880
- Webb, A. M., Reynolds, C., Chen, W., Reeve, H., Iliescu, D.-A., Lujan, M., & Brown, G. (2020). To Ensemble or Not Ensemble: When does End-To-End Training Fail? In ECML PKDD 2020 (pp. 1–16). Retrieved from http://www.cs.man.ac.uk/~gbrown/publications/ecml2020webb.pdf

### Bagging
Bagging uses different subsets of features / training points to train an ensemble of classifiers. The classic version of Bagging uses bootstrap samples that lets each base model slightly overfit to their respective portion of the training data leading to a somewhat diverse ensemble. This implementation supports a few variations of bagging. Similar to SKLearn you can choose the fraction of samples with and without bootstrapping. Moreover, you can freeze all but the last layer of each base model. This simulates a form of feature sampling / feature extraction, and should be expanded in the future. Last, there is a "fast" training method which jointly trains the ensemble using poisson weights for each individual classifier . 

References:

- Breiman, L. (1996). Bagging predictors. Machine Learning. https://doi.org/10.1007/bf00058655
- Webb, G. I. (2000). MultiBoosting: a technique for combining boosting and wagging. Machine Learning. https://doi.org/10.1023/A:1007659514849
- Oza, N. C., & Russell, S. (2001). Online Bagging and Boosting. Retrieved from https://ti.arc.nasa.gov/m/profile/oza/files/ozru01a.pdf 

### (Stochastic) Gradient Boosting

Is implemented and runs. Literature research is TBD

### Stochastic Multiple Choice Learning Classifier
As often argued, diversity might be important for ensembles to work well. Stochastic Multiple Choice Learning (SMCL)
enforces diversity by training each expert model on a subset of the training data for which it already works
pretty well. Due to the random initialization each ensemble member is likely to perform better or worse on different
parts of the data and thereby introducing diversity. SMCL enforces this specialization by selecting the best
expert (wrt. to the loss) for each example and then only updating that one expert for that example. All other experts
will never receive that example. 

References:
- Lee, S., Purushwalkam, S., Cogswell, M., Ranjan, V., Crandall, D., & Batra, D. (2016). Stochastic multiple choice learning for training diverse deep ensembles. Advances in Neural Information Processing Systems, 1(Nips), 2127–2135. Retrieved from http://papers.nips.cc/paper/6270-stochastic-multiple-choice-learning-for-training-diverse-deep-ensembles.pdf

### Stacking
Stacking stacks the predictions of each base learner into one large vector and then trains another model on this new
"example" vector. This implementation can be viewed as End2End stacking, in which both - the base models as well as
the combinator model - are trained in an end-to-end fashion. 
    
References:

- Wolpert, D. (1992). Stacked Generalization ( Stacking ). Neural Networks.
- Breiman, L. (1996). Stacked regressions. Machine Learning. https://doi.org/10.1007/bf00117832

## Training an ensemble

I losely follow the conventions and API established by SKLearn. However, I do not fully support the SKLearn API and I should really 
consider if skorch (https://github.com/skorch-dev/skorch) might be intestring here. When implementing this code, I was not sure 
how pytorch would handel copies of models and therefore decided to use a combination of functions and dictionaries for generating new 
models / setting parameters.



optimizer, 
scheduler, 
loss_function, 
base_estimator, 
training_file="training.jsonl",
transformer = None,
pipeline = None,
seed = None,
verbose = True, 
out_path = None, 
x_test = None, 
y_test = None, 
eval_test = 5,
store_on_eval = False

n_estimators = 5, combination_type = "average"

## To Do 

- AdaBoost from SKLearn can be used. 
    - TODO Use skorch? (https://github.com/skorch-dev/skorch)
- Pretrained
- LazyEnsembleClassifier
- Soft DT
- Hard DT with SGD
