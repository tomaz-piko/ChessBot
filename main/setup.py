from setuptools import setup
from Cython.Build import cythonize
import numpy

files = [
    "main/gameimage/c.pyx",
    "main/mcts/c.pyx"
]

setup(
    name='Hello world app',
    ext_modules=cythonize(files, annotate=True),
    include_dirs=[numpy.get_include()],
)