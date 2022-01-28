
from setuptools import setup
from setuptools import find_packages

setup(name='softmaxkmeans',
      version='0.1',
      description='Softmax k-means',
      author='Sibylle Hess',
      author_email='s.c.hess@tue.nl',
      #packages=['train','train.models','one-pixel-attack'],
      packages=find_packages(include=["train.*"]),
      install_requires=['numpy', 'scipy'],
      zip_safe=False)
