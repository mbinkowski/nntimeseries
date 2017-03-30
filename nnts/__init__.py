"""
Created on Tue Nov 29 17:59:55 2016

@author: mbinkowski

Initialization file.
"""

import os 
from .config import WDIR

for directory in ['logs', 'results', 'data', 'tensorboard']:
    if directory not in os.listdir(WDIR):
        os.mkdir(WDIR + directory)

__all__ = ['artificial', 'household', 'keras_utils', 'utils', 'config', 'models']