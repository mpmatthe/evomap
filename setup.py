#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# evomap
# 
# This file is part of the evomap project: https://github.com/mpmatthe/evomap
# 
# Copyright (c) 2024, Maximilian Matthe, Daniel M. Ringel, Bernd Skiera
# 
# Licensed under the MIT License (see LICENSE for details)
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages
import urllib.request

# Function to extract version number from the __init__.py file
def get_version():
    with open("src/evomap/__init__.py", "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split('=')[1]
                version = version.replace("'", "").replace('"', "").strip()
                return version

# Fetch the long description from the remote README
upstream_url = 'https://raw.githubusercontent.com/mpmatthe/evomap/main/README.md'
response = urllib.request.urlopen(upstream_url)
long_description = response.read().decode('utf-8')

setup(
    name='evomap',
    version=get_version(),
    description='A Python Toolbox for Mapping Evolving Relationship Data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mpmatthe/evomap',
    author='Maximilian Matthe, Daniel M. Ringel, Bernd Skiera',
    author_email='mpmatthe@iu.edu',
    license='MIT',
    packages=find_packages('src'),  # Automatically find packages in the src directory
    package_dir={'': 'src'},
    include_package_data=True,  # Include package data specified in MANIFEST.in
    package_data={
        'evomap.data': ['*.csv', '*.xlsx', '*.npy', '__init__.py'],
    },
    install_requires=[
        'matplotlib',
        'numba',
        'numpy',
        'pandas',
        'scipy',
        'seaborn',
        'statsmodels',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',
    keywords='mapping, data analysis, network analysis, evolving data relationships',
)
