"""
Configuration file. User may set the working directory variable WDIR below to 
the directory of his choice.
"""
import sys, os

SEP = os.path.sep
WDIR = SEP.join(os.path.abspath(__file__).split(SEP)[:-2] + [''])
print('Working directory: ' + repr(WDIR))

import sys 
sys.setrecursionlimit(2000) # to alleviate some problems with model saving.