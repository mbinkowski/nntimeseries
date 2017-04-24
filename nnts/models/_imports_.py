# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:19:40 2017

@author: mbinkowski
"""

import sys, os

sep = os.path.sep
sys.path.append(sep.join(os.path.abspath(__file__).split(sep)[:-3]))
print(sys.path)
import nnts
from nnts._imports_ import *
#nnts.config.WDIR = path + '\\'