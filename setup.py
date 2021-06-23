#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
enseisro - a Python package to carry out inversions
for radial differential rotaion using ensemble astero-
seismology using stars observed by PLATO mission.
:copyright:
    Srijan B. Das (sbdas@Princeton.EDU), 2021
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)
'''
# Importing setuptools monkeypatches some of distutils commands so things like
# 'python setup.py develop' work. Wrap in try/except so it is not an actual
# dependency. Inplace installation with pip works also without importing
# setuptools.

import os
import sys
import math
import argparse
from setuptools import setup
from setuptools import find_packages
from setuptools.command.test import test as TestCommand


setup(
    name='enseisro',
    version='0.1.',
    packages=find_packages("."), # Finds every folder with __init__.py in it. (Hehe)
    install_requires=[
        "numpy", "matplotlib", "scipy", "wigner", "py3nj"
    ],
)
