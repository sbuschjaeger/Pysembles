# PyTorch Ensembles 

This repository contains code for training ensembles with PyTorch. These are mostly straigt-forward implementations 
without much optimizations. I tried to stick as close as possible to the original papers to provide some references. 
For more details on each method please have a look at the header of the respective implementation.
Currently supported are:

- Generalized Negative Correlation Learning
- End2EndLearning
- Bagging
- (Stochastic) Gradient Boosting
- Stochastic Multiple Choice Learning Classifier
- Stacking

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

- Check formulation of GNLC Classifier and maybe add specialized version of NCL_MSE and NCL_CrossEntropy
- AdaBoost from SKLearn can be used. 
    - TODO Use skorch? (https://github.com/skorch-dev/skorch)
- Pretrained
- LazyEnsembleClassifier
- Soft DT
- Hard DT with SGD
- BaseModels 
- Implement https://github.com/dvornikita/fewshot_ensemble