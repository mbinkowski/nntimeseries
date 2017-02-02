# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:57:01 2016

@author: mbinkowski
"""
from __init__ import *

def getXy(filepath, return_stats=False):
    with open(filepath, 'rb') as f:
        XX, names = pickle.load(f)
    """ XX info:
       -1 target price (change)
       -2 abs target price
       -3 target time
       -4 forward time diff
       -5 time
       -6 price
       -7 dir
       0: -8 sources (0-1 for each class)   
    """    
    XX = np.asarray(XX, dtype=np.float32)
    XX = XX[-8:, :, :]   
    N = XX.shape[1]//8
    XX = np.concatenate([XX[:, i*N: (i+1)*N, :] for i in range(8)], axis=0)            
    n_train = int(XX.shape[1]*.8)
    n_all = XX.shape[1]
    print((XX.shape, n_train, n_all))
    Xmeans = XX[:, :n_train, :].mean(axis=1, keepdims=True).mean(axis=0, keepdims=True)
    Xstds = np.sqrt((XX[:, :n_train, :].std(axis=1, keepdims=True)**2).mean(axis=0, keepdims=True)).clip(1e-5, np.inf)
    Xmeans[:, :, -6] = 0#Xmeans[:, :, -6]
    Xmeans[:, :, -2] = 0
    Xstds[:, : ,-6] = Xstds[:, :, -1]
    Xstds[:, :, -2] = Xstds[:, :, -1]
    XX = (XX - Xmeans)/Xstds
    outputs = [XX[:, :, -1:] > 0, XX[:, :, -1:] <= 0, XX[:, :, -1:], XX[:, :, -2:-1] - XX[:, :, -1:]]
    y = np.concatenate(outputs, axis=2)
    # y = XX[:, :, -1:]
    X = XX[:, :, :-4]
    div = ((XX[:, :, -1] - XX[:, : -1].mean())**2).mean()*(Xstds[:, :, -1]/Xstds[:, :, -2])**2   
    if return_stats:
        return X, y, n_train, n_all, div, Xstds, Xmeans
    return X, y, n_train, n_all, div
    
def list_of_param_dicts(param_dict):
    vals = list(prod(*[v for k, v in param_dict.items()]))
    keys = list(prod(*[[k]*len(v) for k, v in param_dict.items()]))
    return [dict([(k, v) for k, v in zip(key, val)]) for key, val in zip(keys, vals)]

#def LR_class_gen(X, y, l, r, length):
#    order = np.random.permutation(np.arange(l, r - length))
#    sh = (X.shape[0], X.shape[2]*length)
#    while True:
#        i = 0
#        while i < len(order):
#            j = order[i]
#            yield X[:, j: j + length, :].reshape(sh), y[:, j + length - 1, :2]
#            i += 1
#
def LR_regr_gen(X, y, l, r, length):
    order = np.random.permutation(np.arange(l, r - length))
    sh = (X.shape[0], X.shape[2]*length)
    while True:
        i = 0
        while i < len(order):
            j = order[i]
            xxx = X[:, j: j + length, :].copy()
            xxx[:, :, -2] -= y[:, j + length - 1: j + length, 3]
            yield xxx.reshape(sh), y[:, j + length - 1, 2:3]
            i += 1
#            
#def CNN_class_gen(X, y, l, r, length):
#    order = np.random.permutation(np.arange(l, r - length))
#    while True:
#        i = 0
#        while i < len(order):
#            j = order[i]
#            yield X[:, j: j + length, :], y[:, j + length - 1, :2]
#            i += 1

def CNN_regr_gen(X, y, l, r, length):
    order = np.random.permutation(np.arange(l, r - length))
    while True:
        i = 0
        while i < len(order):
            j = order[i]
            xxx = X[:, j: j + length, :].copy()
            xxx[:, :, -2] -= y[:, j + length - 1: j + length, 3]
            yield xxx, y[:, j + length - 1, 2:3]
            i += 1
            
#def VI_class_gen(X, y, l, r, length):
#    order = np.random.permutation(np.arange(l, r - length))
#    while True:
#        i = 0
#        while i < len(order):
#            j = order[i]
#            yieldX = {'value_input': X[:, j: j + length, -2], 
#                   'importance_input': X[:, j: j + length, :],
#                   'shift_input': X[:, j: j + length, :-2]}
#            yieldY = {'main_output': y[:, j + length - 1, :2],
#                      'value_output': np.concatenate([y[:, j + length - 1, 2:]]*length, axis=-1)}
#            yield yieldX, yieldY
#            i += 1
            
def VI_regr_gen(X, y, l, r, length):
    order = np.random.permutation(np.arange(l, r - length))
    while True:
        i = 0
        while i < len(order):
            j = order[i]
            xxx = X[:, j: j + length, :].copy()
            xxx[:, :, -2] -= y[:, j + length - 1: j + length, 3]
            yieldX = {'value_input': xxx[:, :, -2], 
                   'importance_input': xxx,
                   'shift_input': xxx[:, :, :-2]}
            yieldY = {'main_output': y[:, j + length - 1, 2:3],
                      'value_output': np.concatenate([y[:, j + length - 1, 2:3]]*length, axis=-1)}
            yield yieldX, yieldY
            i += 1