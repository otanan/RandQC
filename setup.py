#!/usr/bin/env python3
"""
====================================
Filename:       setup.py
Author:         Jonathan Delgado 
Description:    Setup for pip installing RandQC.
====================================
"""

from setuptools import setup, find_packages

setup(
    name='RandQC',
    version='0.1.0.1',
    description='Random Quantum Circuits.',
    url='https://github.com/otanan/RandQC',
    author='Jonathan Delgado',
    author_email='jonathan.delgado@uci.edu',

    # packages=find_packages(),
    packages=[
        'randqc',
        'randqc.tools',
    ],
    install_requires=[
        # External packages
        'numpy',
        'scipy',
        # Dependency for qutip
        'Cython',
        'qutip',
        'matplotlib',
        # Entropy
        'scikit-image',
        # IO
        'h5py',
    ],
)