from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "main.gameimage.c",
        sources=["main/gameimage/c.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "main.mcts.c",
        sources=["main/mcts/c.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "main.cppchess.__init__",
        sources=["main/cppchess/__init__.pyx"],
        language="c++",
        extra_compile_args=["-std=c++20", "-O3"],
    )
]

setup(
    name="cy",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)