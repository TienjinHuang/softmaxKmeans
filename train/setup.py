from setuptools import setup  
setup(name='softmaxkmeans_models',       
  version='0.1',       
  description='Softmax k-means',       
  author='Sibylle Hess',       
  author_email='s.c.hess@tue.nl',       
  packages=['models'],       
  install_requires=['numpy', 'scipy'],       
  zip_safe=False)
