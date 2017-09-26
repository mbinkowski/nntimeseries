# -*- coding: utf-8 -*-
"""
Import file.
"""

# misc
import os
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
from keras.layers.core import Lambda, RepeatVector, Flatten, Permute
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l1, l2, l1_l2
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
    
from nnts import utils, keras_utils, user, models
from nnts.models import *

