from setuptools import setup, Extension

from Cython.Build import cythonize

import numpy as np

setup(install_requires=['numpy', 'scipy'],
      ext_modules=cythonize([
          Extension("ttide.t_tidec", ["ttide/t_tidec.pyx"],)
      ]),
      include_dirs=[np.get_include()]
)
