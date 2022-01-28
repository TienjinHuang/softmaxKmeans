
from setuptools import setup

setup(name='softmaxkmeans',
      version='0.1',
      description='Softmax k-means',
      author='Sibylle Hess',
      author_email='s.c.hess@tue.nl',
      packages=['train','train.models','one-pixel-attack'],
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
