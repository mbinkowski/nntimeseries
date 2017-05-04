# -*- coding: utf-8 -*-
"""
Created on Thu May  4 17:17:35 2017

@author: mbinkowski
"""

from ._imports_ import *
from .config import WDIR

class QuotesGenerator(utils.Generator):
    def __init__(self, filepath, return_stats=True, scale_by_returns=True, 
                 batch_size=64, train_share=(.7, .8, 1), input_length=1, 
                 output_length=1, verbose=1, limit=np.inf, ):
        with open(filepath, 'rb') as f:
            X, names = pickle.load(f)
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
        batch_n = max(1, batch_size//8)
        XX = np.asarray(XX, dtype=np.float32)
        XX = XX[-8:, :, :]   
        N = XX.shape[1]//batch_n
        XX = np.concatenate([XX[:, i*N: (i+1)*N, :] for i in range(batch_n)], axis=0)            
        n_train = int(XX.shape[1] * train_share[-2])
        n_all = int(XX.shape[1] * train_share[-1])
        print((XX.shape, n_train, n_all))
        Xmeans = XX[:, :n_train, :].mean(axis=1, keepdims=True).mean(axis=0, keepdims=True)
        Xstds = np.sqrt((XX[:, :n_train, :].std(axis=1, keepdims=True)**2).mean(axis=0, keepdims=True)).clip(1e-5, np.inf)
        Xmeans[:, :, -4] = 0
        Xstds[:, :, -4] = 1
    #    Xmeans[:, :, -1] = 0 ## comment to first NeX results
        if scale_by_returns:
            Xmeans[:, :, -6] = 0 #Xmeans[:, :, -6]
            Xstds[:, : ,-6] = Xstds[:, :, -1]
        else:
            Xstds[:, :, -1] = Xstds[:, :, -6]
        Xmeans[:, :, -2] = Xmeans[:, :, -6]
        Xstds[:, :, -2] = Xstds[:, :, -6]
        XX = (XX - Xmeans)/Xstds
        outputs = [XX[:, :, -1:] > 0, XX[:, :, -1:] <= 0, XX[:, :, -1:],# 0-2 returns
                   XX[:, :, -2:-1],                                     # 3 price
                   XX[:, :, -2:-1] - XX[:, :, -1:],                     # 4 previous price
                   XX[:, : ,-4:-3]]                                     # 5 time diff
#        y = np.concatenate(outputs, axis=2)
#        X = XX[:, :, :-4]
        X = np.concatenate(outputs + [XX[:, :, :-4]], axis=2)
        self.div = ((XX[:, :, -1] - XX[:, : -1].mean())**2).mean()*(Xstds[:, :, -1]/Xstds[:, :, -2])**2 
        super(QuotesGenerator, self).__init__(
            self, X, 
            train_share=train_share, input_length=input_length, 
            output_length=output_length, verbose=verbose, limit=limit, 
            batch_size=batch_size, excluded=[], diffs=scale_by_returns)
    
    def asarray(self):
        return self.X
        
    def get_target_col_ids
        

        
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:57:01 2016

@author: mbinkowski
"""
from __init__ import *

def getXy(filepath, return_stats=False):
    with open(filepath, 'rb') as f:
        XX, names = pickle.load(f)
    XX = np.asarray(XX, dtype=np.float32)
    XX = XX[-8:, :, :]   
    N = XX.shape[1]//8
    XX = np.concatenate([XX[:, i*N: (i+1)*N, :] for i in range(8)], axis=0)            
    n_train = int(XX.shape[1]*.8)
    n_all = XX.shape[1]
    print((XX.shape, n_train, n_all))
    Xmeans = XX[:, :n_train, :].mean(axis=1, keepdims=True).mean(axis=0, keepdims=True)
    Xstds = np.sqrt((XX[:, :n_train, :].std(axis=1, keepdims=True)**2).mean(axis=0, keepdims=True)).clip(1e-5, np.inf)
    Xmeans[:, :, -2] = Xmeans[:, :, -6]
    Xmeans[:, :, -1] = 0
    Xmeans[:, :, -4] = 0
    Xstds[:, :, -2] = Xstds[:, :, -6]
    Xstds[:, :, -1] = 1
    Xstds[:, :, -4] = 1
    XX = (XX - Xmeans)/Xstds
    outputs = [XX[:, :, -1:] > 0, XX[:, :, -1:] <= 0, XX[:, :, -2:-1]]
    if return_stats:
        outputs += [XX[:, :, -1:], XX[:, :, -4:-3]]
    y = np.concatenate(outputs, axis=2)
    # y = XX[:, :, -1:]
    X = XX[:, :, :-4]
    div = ((XX[:, :, -1] - XX[:, : -1].mean())**2).mean()*(Xstds[:, :, -1]/Xstds[:, :, -2])**2   
    if return_stats:
        return X, y, n_train, n_all, div, Xstds, Xmeans
    return X, y, n_train, n_all, div

def getXy2(filepath, return_stats=True, scale_by_returns=True, batch_size=64):
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
    batch_n = max(1, batch_size//8)
    XX = np.asarray(XX, dtype=np.float32)
    XX = XX[-8:, :, :]   
    N = XX.shape[1]//batch_n
    XX = np.concatenate([XX[:, i*N: (i+1)*N, :] for i in range(batch_n)], axis=0)            
    n_train = int(XX.shape[1]*.8)
    n_all = XX.shape[1]
    print((XX.shape, n_train, n_all))
    Xmeans = XX[:, :n_train, :].mean(axis=1, keepdims=True).mean(axis=0, keepdims=True)
    Xstds = np.sqrt((XX[:, :n_train, :].std(axis=1, keepdims=True)**2).mean(axis=0, keepdims=True)).clip(1e-5, np.inf)
    Xmeans[:, :, -4] = 0
    Xstds[:, :, -4] = 1
#    Xmeans[:, :, -1] = 0 ## comment to first NeX results
    if scale_by_returns:
        Xmeans[:, :, -6] = 0 #Xmeans[:, :, -6]
        Xstds[:, : ,-6] = Xstds[:, :, -1]
    else:
        Xstds[:, :, -1] = Xstds[:, :, -6]
    Xmeans[:, :, -2] = Xmeans[:, :, -6]
    Xstds[:, :, -2] = Xstds[:, :, -6]
    XX = (XX - Xmeans)/Xstds
    outputs = [XX[:, :, -1:] > 0, XX[:, :, -1:] <= 0, XX[:, :, -1:],# 0-2 returns
               XX[:, :, -2:-1],                                     # 3 price
               XX[:, :, -2:-1] - XX[:, :, -1:],                     # 4 previous price
               XX[:, : ,-4:-3]]                                     # 5 time diff
    y = np.concatenate(outputs, axis=2)
    X = XX[:, :, :-4]
    div = ((XX[:, :, -1] - XX[:, : -1].mean())**2).mean()*(Xstds[:, :, -1]/Xstds[:, :, -2])**2   
    if return_stats:
        return X, y, n_train, n_all, div, Xstds, Xmeans
    return X, y, n_train, n_all, div    
    
    
def xy_validator(xx, yy, tlimit=600, std=.5):
    zz = yy[:, :, -1].max(axis=1) < tlimit
    zz2 = (yy[:, -1, 2] < 3*std) & (yy[:, -1, 2] > -3*std)
    return xx, yy[:,:,:], (zz & zz2)

def validator0(xx, yy, std=0):
    return xx, yy, [True]*xx.shape[0]    
    
def gen(X, y, l, r, length, f, recurrent=False):
    if recurrent:
        order = np.arange(l, r - length)
    else:
        order = np.random.permutation(np.arange(l, r - length))
    while True:
        i = 0
        while i < len(order):
            j = order[i]
            yield f(X[:, j: j + length, :], y[:, j: j + length, :])
            i += 1    

def vgen(X, y, l, r, length, func, validator=validator0):
    order = np.random.permutation(np.arange(l, r - length))
    Xl, yl = [], []
    N = X.shape[0]
    std = y[:, :, 2].std()
    while True:
        i = 0
        excl = 0
        while i < len(order):
            while len(Xl) < N:                
                j = order[i]
                xx, yy, zz = validator(X[:, j: j + length, :], y[:, j: j + length, :], std=std)
                for k in range(N):
                    if zz[k]:
                        Xl.append(xx[k: k+1])
                        yl.append(yy[k: k+1])
                    else:
                        excl += 1
                i += 1
                if i >= len(order):
                    i = 0
                    print('Excluded: ' + repr(excl))
            yield func(np.concatenate(Xl[:N], axis=0), np.concatenate(yl[:N], axis=0))
            Xl, yl = Xl[N:], yl[N:]
        print('Excluded: ' + repr(excl))


def LR_class(xx, yy):
    sh = (xx.shape[0], xx.shape[1]*xx.shape[2])
    return xx.reshape(sh), yy[:, -1, :2]        

def LR_regr(xx, yy):
    sh = (xx.shape[0], xx.shape[1]*xx.shape[2])
    return xx.reshape(sh), yy[:, -1, 3:4]    

def LR_ret_regr(xx, yy):
    sh = (xx.shape[0], xx.shape[1]*xx.shape[2])
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    return xxx.reshape(sh), yy[:, -1, 2:3]
    
def LR_ret_class(xx, yy):
    sh = (xx.shape[0], xx.shape[1]*xx.shape[2])
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    return xxx.reshape(sh), yy[:, -1, :2]
    
def CNN_class(xx, yy):           
    return xx, yy[:, -1, :2]            

def CNN_regr(xx, yy):           
    return xx, yy[:, -1, 3:4]  

def CNN_ret_class(xx, yy):
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    return xxx, yy[:, -1, :2] 
    
def CNN_ret_regr(xx, yy):
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    return xxx, yy[:, -1, 2:3]     

def VI_class(xx, yy):            
    yieldX = {'value_input': xx[:, :, -2], 
           'importance_input': xx,
           'shift_input': xx[:, :, :-2]}
    yieldY = {'main_output': yy[:, -1, :2],
              'value_output': np.concatenate([yy[:, -1, 3:4]]*xx.shape[1], axis=-1)}
    return yieldX, yieldY

def VI_regr(xx, yy):           
    yieldX = {'value_input': xx[:, :, -2], 
           'importance_input': xx,
           'shift_input': xx[:, :, :-2]}
    yieldY = {'main_output': yy[:, -1, 3:4],
              'value_output': np.concatenate([yy[:, -1, 3:4]]*xx.shape[1], axis=-1)}
    return yieldX, yieldY            
    
def VI_ret_regr(xx, yy):
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    yieldX = {'value_input': xxx[:, :, -2], 
              'importance_input': xxx,
              'shift_input': xxx[:, :, :-2]}
    yieldY = {'main_output': yy[:, -1, 2:3],
              'value_output': np.concatenate([yy[:,-1, 2:3]]*xx.shape[1], axis=-1)}
    return yieldX, yieldY   
    
def VI_ret_class(xx, yy):
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    yieldX = {'value_input': xxx[:, :, -2], 
              'importance_input': xxx,
              'shift_input': xxx[:, :, :-2]}
    yieldY = {'main_output': yy[:, -1, :2],
              'value_output': np.concatenate([yy[:,-1, 2:3]]*xx.shape[1], axis=-1)}
    return yieldX, yieldY 

def VI3_ret_class(xx, yy):
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    yieldX = {'value_input': xxx[:, :, -2], 
              'importance_input': xxx,
              'shift_input': xxx[:, :, :-2]}
    yieldY = {'main_output': yy[:, -1, :2],
              'value_output': np.concatenate(yy[:,-1, 2:3], axis=-1)}
    return yieldX, yieldY 
    
def RNN_ret_regr(xx, yy):
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    return xxx, yy[:, :, 2:3]
    
def RNN_ret_class(xx, yy):
    xxx = xx.copy()
    xxx[:, :, -2] -= yy[:, -1:, 4]
    return xxx, yy[:, :, :2]