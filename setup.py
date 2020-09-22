from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

modules= []

# TODO WE ASSUME THAT CUDA IS AVAILABLE 
if CUDA_HOME:
    modules.append(
        CUDAExtension('binarization', [
                  'deep_ensembles_v2/cuda/binarization/binarization.cpp',
                  'deep_ensembles_v2/cuda/binarization/binarize_cuda.cu',
            ]),
    )

setup(name='deep_ensembles_v2',
      version='0.1',
      description='Common ensemble approaches for PyTorch models and some soft decision tree approaches. Also includes some code for training binarized neural networks. Many thanks at Mikail Yayla (mikail.yayla@tu-dortmund.de) for providing CUDA kernels for BNN training. He maintains a more evolved repository on BNNs - check it out at https://github.com/myay/BFITT',
      url='https://github.com/sbuschjaeger/deep_ensembles_v2/',
      author=u'Sebastian Buschjäger',
      author_email='sebastian.buschjaeger@tu-dortmund.de',
      license='MIT',
      packages=['deep_ensembles_v2'],
      zip_safe=False,
      install_requires = [
            'torch'
      ],
      ext_modules=modules,
      cmdclass={
            'build_ext': BuildExtension
      }
)