# -*- coding: utf-8 -*-
"""
Imports file.
"""

import sys, os

sep = os.path.sep
sys.path.append(sep.join(os.path.abspath(__file__).split(sep)[:-3]))
print(sys.path)
import nnts
from nnts._imports_ import *
#nnts.config.WDIR = path + '\\'