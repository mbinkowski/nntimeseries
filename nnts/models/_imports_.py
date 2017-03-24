# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 12:19:40 2017

@author: mbinkowski
"""

import sys, os

sys.path.append('\\'.join(os.path.abspath(__file__).split('\\')[:-3]))
print(sys.path)
import nnts
from nnts._imports_ import *
#nnts.config.WDIR = path + '\\'