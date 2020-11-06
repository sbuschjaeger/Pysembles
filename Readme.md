# PyTorch Ensembles 

This repository contains code for training ensembles with PyTorch and was implemented for our paper "Generalized Negative Correlation Learning for Deep Ensembling" (https://arxiv.org/abs/2011.02952). These are mostly straightforward implementations with some optimizations. However, I tried to stick as close as possible to the original papers to provide some references. For more details on each method please have a look at the header of the respective implementation.
Currently, supported are:

- (Generalized) Negative Correlation Learning
- Bagging
- (Stochastic) Gradient Boosting
- Snapshot Ensembles
- Stochastic Multiple Choice Learning
- End2End Ensembles
- Stacking

For reference, we also provide some common architectures as base learners, including

- DenseNet
- EfficientNet
- MobileNetV3
- ResNet and SimpleResNet
- VGG

Last, this code also contains some random stuff including

- Single models with an sklearn-like API supporting (a primitive version of) pipelining
- Autoencoders
- Soft Decision Trees
- Binarized Neural Networks. Many thanks to Mikail Yayla (mikail.yayla@tu-dortmund.de) for providing CUDA kernels for BNN training. He maintains a more evolved repository on BNNs - check it out at https://github.com/myay/BFITT

**Note:** I was lazy during the implementation and used `cuda()` quite a lot in various places. Thus, it is likely that this code will not run a cpu immediately. 

## How to use this code

Please have a look at https://github.com/sbuschjaeger/gncl which explains the experiments more detailed. 
We loosely follow the conventions and API established by SKLearn. However, I do not fully support the SKLearn API and I should really 
consider if skorch (https://github.com/skorch-dev/skorch) might be interesting here. When implementing this code, I was not sure 
how pytorch would handle copies of models and therefore decided to use a combination of functions and dictionaries for generating new 
models / setting parameters. I tried to comment as much code as possible, so please have a look at each individual method for more information on parameters.


### To Do 
- [ ] Check if `sample_weights` is respected in all learners
- [ ] General cleanup
- [ ] Add more comments
- [ ] Use skorch? (https://github.com/skorch-dev/skorch)
- [ ] Check for cuda and call `cuda()` appropriately

## Citing our Paper

    @misc{buschjäger2020generalized,
        title={Generalized Negative Correlation Learning for Deep Ensembling}, 
        author={Sebastian Buschjäger and Lukas Pfahler and Katharina Morik},
        year={2020},
        eprint={2011.02952},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }

## Acknowledgments 
Special Thanks goes to [Lukas Pfahler](https://github.com/Whadup) (lukas.pfahler@tu-dortmund.de) who probably found more bugs in my code than lines I wrote. 