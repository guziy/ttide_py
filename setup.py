from setuptools import setup, Extension

from Cython.Build import cythonize

import numpy as np

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='ttide',
      version='0.3.4',
      description='Python distribution of the MatLab package TTide.',
      long_description=readme(),
      url='https://github.com/moflaher/ttide_py',
      author='Mitchell O\'Flaherty-Sproul',
      author_email='073208o@acadiau.ca',
      license='MIT',
      packages=['ttide'],
      package_data={'ttide': ['data/*.nc']},
      zip_safe=False, install_requires=['numpy', 'scipy'],
      ext_modules=cythonize([
          Extension("ttide.t_tidec", ["ttide/t_tidec.pyx"],)
      ]),
      include_dirs=[np.get_include()]
)
