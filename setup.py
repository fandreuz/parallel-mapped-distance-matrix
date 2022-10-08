#!/usr/bin/env python
from setuptools import setup

setup(name='parallel-mapped-distance-matrix',
    version='0.1',
    description='Methods for the computation of the mapped distance matrix',
    author='Francesco Andreuzzi',
    author_email='andreuzzi.francesco@gmail.com',
    packages=['pmdm'],
    install_requires=['numpy', 'scipy', 'numba', 'csr'],
    license='MIT',
)
