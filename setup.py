from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

modules= []

# TODO WE ASSUME THAT CUDA IS AVAILABLE 
# if CUDA_HOME:
modules.append(
      CUDAExtension('binarization', [
            'pysembles/cuda/binarization/binarization.cpp',
            'pysembles/cuda/binarization/binarize_cuda.cu',
      ]),
)

setup(name='Pysembles',
      version='0.2',
      description='Common ensemble approaches for PyTorch models and some soft decision tree approaches. Also includes some code for training binarized neural networks. Many thanks at Mikail Yayla (mikail.yayla@tu-dortmund.de) for providing CUDA kernels for BNN training. He maintains a more evolved repository on BNNs - check it out at https://github.com/myay/BFITT',
      url='https://github.com/sbuschjaeger/deep_ensembles_v2/',
      author=u'Sebastian Buschjäger',
      author_email='sebastian.buschjaeger@tu-dortmund.de',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires = [
            'torch'
      ],
      ext_modules=modules,
      cmdclass={
            'build_ext': BuildExtension
      }
)