"""
Initialization file.
"""
# self & directory structure
from .config import WDIR

import os
for directory in ['logs', 'results', 'data', 'tensorboard']:
    if directory not in os.listdir(WDIR):
        os.mkdir(WDIR + directory)

from . import utils, keras_utils, artificial
from . import models

__all__ = ['artificial', 'household', 'keras_utils', 'utils', 'config', 'models']