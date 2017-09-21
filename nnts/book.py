# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:56:31 2017

@author: mbinkowski
"""
from ._imports_ import *
from .config import WDIR
from keras import backend as K

class BookGenerator(utils.Generator):
    def __init__(self, filename, 
                 train_share=(.7, .8, 1.0), input_length=1024, output_length=256,
                 verbose=1, limit=np.inf, batch_size=64, diffs=False):
        book = pd.read_pickle(os.path.join(WDIR, filename))
        excluded = [c for c in book.columns if 'time' in c] + ['date', 'count', 'current_ask', 'current_bid', 'ask', 'bid', 'index', 'mean', 'level_0'] 
#        book.reset_index(inplace=True)
        self._vcols = [c for c in book.columns if ('bid' in c) or ('ask' in c)]
        self._ccols = [c for c in book.columns if 'count' in c]
        super(BookGenerator, self).__init__(book, train_share=train_share, 
                                                  input_length=input_length, 
                                                  output_length=output_length, 
                                                  verbose=verbose, limit=limit,
                                                  batch_size=batch_size,
                                                  excluded=excluded,
                                                  diffs=diffs)                   
        self._tcids = self.get_target_col_ids()
        self._ccids = self.get_target_col_ids(cols=self._ccols)
        self._vcids = self.get_target_col_ids(cols=self._vcols)
        self.__scale()
        self.X['mean'] = .5 * (self.X['current_bid'] + self.X['current_ask'])
        
    def _get_ith_sample(self, i):
        ref = self.X.loc[i].copy()
        ref[self._vcols] = ref['mean']
        ref[self._ccols] = 0
        return np.array(self.X.loc[i - self.input_length: i + self.output_length - 1, self.cols] - ref[self.cols])
    
    def get_target_col_ids(self, ids=True, cols='default'):
        if cols in ['all', 'default']:
            cols = ['best_ask', 'best_bid']
        assert type(cols) == list
        return [(i if ids else c) for i, c in enumerate(self.cols) if c in cols]
        
    def _scale(self):
        pass
    
    def __scale(self):
        arr = self.asarray()
        varr = arr[:self.n_train, self._tcids]
        tarr = arr[:self.n_train, -1]
        l = int(np.sqrt(self.input_length))
        self.X[self._vcols] /= (varr[l:] - varr[:-l]).std()
        self.X[self._ccols] /= self.X[self._ccols].std().mean() 
        self.X['seconds'] /= (tarr[l:] - tarr[:-l]).std()
        
    def make_io_func(self, io_form, cols='default', input_cols=None):
        il = self.input_length
        cols = self.get_target_col_ids(cols)
        if 'exp_time' in io_form:
            times, t = [], self.output_length
            while t >= 1:
                times.append(il - 1 + t)
                t = int(t/2)
            if io_form == '0+exp_time':
                times.append(il)
            times = times[::-1]
            def regr(x):
                return (x[:, :il, :] if (input_cols is None) else x[:, : il, input_cols],
                        x[:, times, :][:, :, cols])
            return regr
        else:
            return super(BookGenerator, self).make_io_func(io_form, cols, input_cols)
 
            
def def_pnl(i):       
    def pnl(y_true, y_pred):
#        print('y_pred[:, i: i+1, 1] - y_true[:, :1, 0] = ' + repr(y_pred[:, i, 1] - y_true[:, 0, 0]))
#        print('y_pred[:, i: i+1, 1] - y_true[:, :1, 0] > 0 : ' + repr(y_pred[:, i: i+1, 1] - y_true[:, :1, 0] > 0))
        Iup = K.cast(y_pred[:, i, 1] - y_true[:, 0, 0] > 0, K.floatx()) #b^ > a
        Idown = K.cast(y_pred[:, i, 0] - y_true[:, 0, 1] < 0, K.floatx()) #a^ < b
        Pup = y_true[:, i, 1] - y_true[:, 0, 0]
        Pdown = y_true[:, 0, 1] - y_true[:, i, 0]
        return K.sum(Iup * Pup + Idown * Pdown)
    return pnl

    
def pnl_loss(y_true, y_pred):
    Iup = K.sigmoid(-y_pred[1:, 1] + y_true[:1, 0]) #b^ > a
    Idown = K.sigmoid(y_pred[1:, 0] - y_true[:1, 1]) #a^ < b
    Pup = y_true[1:, 1] - y_true[:1, 0]
    Pdown = y_true[:1, 1] - y_true[1:, 0]
    return -K.sum(Iup * Pup + Idown * Pdown)  
    
    
def def_pnl_loss_L2(alpha):
    def pnl_loss_L2(y_true, y_pred):
        return alpha * pnl_loss(y_true, y_pred) + (1 - alpha) * K.mean(K.square(y_true - y_pred))
    return pnl_loss_L2
           