"""
Initialization file.
"""

import os 
from .config import WDIR

for directory in ['logs', 'results', 'data', 'tensorboard']:
    if directory not in os.listdir(WDIR):
        os.mkdir(WDIR + directory)

__all__ = ['artificial', 'household', 'keras_utils', 'utils', 'config', 'models']