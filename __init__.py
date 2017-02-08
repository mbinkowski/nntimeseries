# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:59:55 2016

@author: mbinkowski
"""

# settings
import os 
os.chdir('C://Users//mbinkowski//cdsol-r-d.cluster//cdsol-r-d.machine_learning_studies//nntimeseries')
for directory in ['logs', 'results']:
    if directory not in os.listdir(os.getcwd()):
        os.mkdir(directory)
import sys 
sys.setrecursionlimit(2000)

# misc
import numpy as np
import pandas as pd
import datetime as dt
import time
import imp
import itertools
from itertools import product as prod
import pickle
import dill
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
    import tensorflow as tf
else:
    # theano
    import theano
    from theano import tensor as T
    from theano.tensor.nnet import conv2d
    ###############################
    theano.config.blas.ldflags = ''
    ###############################

# contrib
#from quotebook_utils import *
from keras_utils import *
from utils import *
#from artificial_utils import *
#from data import datum