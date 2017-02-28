"""
Created on Tue Nov 29 17:59:55 2016

@author: mbinkowski

Initialization file. Set the working directory variable WDIR below to 
the directory where the .py files are stored.
"""

import os 

#WDIR = os.environ['WORK'] + '//nntimeseries/'
WDIR = 'C://Users//mbinkowski//cdsol-r-d.cluster//cdsol-r-d.machine_learning_studies//nntimeseries/'

os.chdir(WDIR)

for directory in ['logs', 'results', 'data']:
    if directory not in os.listdir(os.getcwd()):
        os.mkdir(directory)
import sys 
sys.setrecursionlimit(2000) # to alleviate some problems with model saving.

# misc
import numpy as np
import pandas as pd
import datetime as dt
import time
import imp
import itertools
from itertools import product as prod
import pickle
import string
import datetime

# plotting
import matplotlib
import matplotlib.pyplot as plt
from IPython import display
import seaborn as sns
  
# keras
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout, Reshape, Input, merge, LocallyConnected1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Merge, Lambda, RepeatVector, Flatten, Permute
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l1, l2, l1l2
from keras.constraints import unitnorm, nonneg, maxnorm
from keras import backend as K

if K._BACKEND == 'tensorflow':
    # tensorflow
    import tensorflow as tf
else:
    # theano
    import theano
    from theano import tensor as T
    from theano.tensor.nnet import conv2d
    ###############################
    theano.config.blas.ldflags = ''
    ###############################


import utils
from keras_utils import *
from model_functions import *

