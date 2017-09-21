# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 18:58:19 2017

@author: mbinkowski
"""

from ._imports_ import *
from .config import WDIR

class LOBSTERGenerator(utils.Generator):
    def __init__(self, filename, keep_lvl=10, _time=.01,
                 train_share=(.8, 1), input_length=100, output_length=1, 
                 verbose=1, limit=np.inf, batch_size=16, diffs=True,
                 chunk=10000):
        self.keep_lvl = keep_lvl
        self.time = _time
        
        self.nameBook = os.path.join(WDIR, filename, 'orderbook.csv')
        self.nameMess = os.path.join(WDIR, filename, 'message.csv')       
        self.file_lvl = int(filename.split('_')[-1])
        
        self.train_share = train_share
        self.batch_size = batch_size
        self.input_length = input_length
        self.output_length = output_length
        self.l = input_length + output_length
        self.verbose = verbose
        self.limit = limit
        self.batch_size = batch_size
        self.diffs = diffs
        self.chunk = chunk
        self.tt = time.time()
        
        self.mess = pd.read_csv(self.nameMess, names=['Time', 'Type', 'ID', 'Size', 'Price', 'Direction'])
        start = 9.5*60*60
        end = 16*60*60
        timeIdx = (self.mess['Time'] >= start) & (self.mess['Time'] <= end)
        self.mess = self.mess[timeIdx]

        self.max_end = (self.mess['Time'] <= self.mess['Time'][self.mess.index[-1]] - .01).sum() - 1

        self.n_train = int((self.max_end * train_share[0] - self.l)/batch_size) * batch_size + self.l
        self.n_valid = self.n_train + int((self.max_end * train_share[1] - self.n_train - self.l)/batch_size) * batch_size + self.l
        if len(train_share) > 2:
            self.n_test = self.n_valid + int((self.max_end * train_share[2] - self.n_valid - self.l)/batch_size) * batch_size + self.l
            self.test = True
        else:
            self.test = False
        
        bookNames = []
        for i in np.arange(1, self.file_lvl + 1):
            bookNames += ['AskP %i' % i, 'AskV %i' % i, 'BidP %i' % i, 'BidV %i' % i]
        self.book = pd.read_csv(self.nameBook, names=bookNames)
        
        Vcols = [c for c in self.book.columns if 'V' in c]
        self.Bmean = self.book.loc[:self.n_valid, Vcols].mean().mean()
        self.Mmean = self.mess.loc[:self.n_valid, 'Size'].mean()
        self.Bstd = np.array(self.book.loc[:self.n_valid, Vcols]).std()
        self.Mstd = self.mess.loc[:self.n_valid, 'Size'].std()
        
        self.book[Vcols] = (self.book[Vcols] - self.Bmean) / (self.Bstd + 1e-6*(self.Bstd == 0))
        self.mess['Size'] = (self.mess['Size'] - self.Mmean) / (self.Mstd + 1e-6*(self.Mstd == 0))
    
        
        self.Bchunks, self.Mchunks = [], []
        self.mess['Size'] = self.mess['Size'] * self.mess['Direction']
        for jj in range(1, 6):
            self.mess['Type%d' % jj] = (self.mess['Type'] == jj)
        for k in range(int(np.ceil(float(self.book.shape[0])/chunk))):
            book0 = self.book.loc[k*chunk: (k+1) * chunk - 1]
            Prange = np.arange(book0['BidP %d' % self.file_lvl].min(), book0['AskP %d' % self.file_lvl].max() + 1, 100)
            chunk_len = min(chunk, self.book.shape[0] - k*chunk)
            Vchunk = pd.DataFrame(np.zeros((chunk_len, len(Prange))), columns=Prange, 
                                  index=np.arange(k*chunk, k*chunk + chunk_len))
            Mchunk = pd.DataFrame(np.zeros((chunk_len, len(Prange))), columns=Prange, 
                                  index=np.arange(k*chunk, k*chunk + chunk_len))
            for lvl in np.arange(1, self.file_lvl + 1):
                Vbid = book0.pivot(columns='BidP %d' % lvl, values='BidV %d' % lvl).fillna(0.0)
                Vchunk[Vbid.columns] -= Vbid
                Vask = book0.pivot(columns='AskP %d' % lvl, values='AskV %d' % lvl).fillna(0.0)
                Vchunk[Vask.columns] += Vask
            MV = self.mess.loc[k*chunk: (k+1) * chunk - 1].pivot(values='Size', columns='Price').fillna(.0)
            Mchunk[MV.columns] = MV
            self.Bchunks.append(Vchunk)
            self.Mchunks.append(Mchunk)
        self.Pmid = ((self.book['AskP 1'] + self.book['BidP 1'])//200 + 1) * 100  
        
            
    def get_target_col_ids(self, cols, ids=True):
        if cols == 'default':
            return np.arange(6, 6 + 2 * self.keep_lvl)
        elif type(cols) == int:
            return np.arange(6 + self.keep_lvl - cols, 6 + self.keep_lvl + cols)
        raise ValueError('cols = ' + repr(cols) + ' not understood')

    def get_dim(self):
        return self.keep_lvl * 2 + 6

    def get_dims(self, cols):
        return self.get_dim(), len(self.get_target_col_ids(cols=cols))
            
    def _get_ith_sample(self, i, return_xy=False):
        assert self.input_length <= i, 'n=%d too large for i=%d' % (self.input_length, i)
        future = i - 1 + (self.mess['Time'][i:] <= self.mess['Time'][i] + self.time).sum()
        iP = np.arange(self.Pmid[i] - self.keep_lvl*100, self.Pmid[i] + self.keep_lvl*100, 100)
        state = np.c_[np.zeros((1, 6)),
                      [np.array(self.Bchunks[i//self.chunk].loc[i, iP])]]
        future_state = np.c_[np.zeros((1, 6)),
                             [np.array(self.Bchunks[future//self.chunk].loc[future, iP])]]
        i_max = i//self.chunk
        i_min = (i - self.input_length + 2)//self.chunk
        i_chunks = [chunk.loc[i - self.input_length + 2: i, iP] for chunk in self.Mchunks[i_min: i_max + 1]]
        i_Types = self.mess.loc[i - self.input_length + 2: i, ['Time'] + ['Type%d' % jj for jj in range(1, 6)]]
        i_Types['Time'] -= i_Types.loc[i, 'Time']
        Messages = np.c_[np.array(i_Types), np.array(pd.concat(i_chunks))]
        
        if self.diffs:
            future_state -= state
        if return_xy:
            return np.nan_to_num(np.r_[Messages, state]), np.nan_to_num(future_state)
        return np.nan_to_num(np.r_[Messages, state, future_state])
        
    
        
        
#    def _get_ith_sample(self, i, return_xy=False):
##        print('getting sample at %dth position, time = %f' % (i, time.time() - self.tt))
#        assert self.input_length <= i, 'n=%d too large for i=%d' % (self.input_length, i)
#        future = i - 1 + (self.mess['Time'][i:] <= self.mess['Time'][i] + self.time).sum()
#        current_state = self.book.loc[i].copy()
#        future_state = self.book.loc[future].copy()
#        updates = np.zeros((self.input_length, 2*self.keep_lvl))
#        future_lvls = np.zeros((1, 2*self.keep_lvl))
#        ttypes = np.zeros((self.input_length, 6))
#        refP = (current_state['AskP 1'] + current_state['BidP 1'])//200 * 100
#        for state, lvl_array in zip([current_state, future_state], [updates[-1], future_lvls[0]]):
#            Pcols = [c for c in state.index if 'P' in c]
#            state[Pcols] = (state[Pcols] - refP)/100 + self.keep_lvl
#            for j in range(1, len(Pcols)//2 + 1):
#                lvl = state['BidP %d' % j]
#                assert lvl - int(lvl) == 0, 'wrong lvl: %f' % lvl
#                if lvl < 0:
#                    break
#                lvl_array[int(lvl)] = -state['BidV %d' % j]
#            for j in range(1, len(Pcols)//2 + 1):
#                lvl = state['AskP %d' % j]
#                assert lvl - int(lvl) == 0, 'wrong lvl: %f' % lvl
#                if lvl >= 2 * self.keep_lvl:
#                    break
#                lvl_array[int(lvl)] = state['AskV %d' % j]   
#        mess0 = self.mess.loc[i - self.input_length + 2: i, :].copy()
#        mess0.loc[:, 'Price'] = np.int32((mess0['Price'] - refP)//100 + self.keep_lvl)
#        mess0.loc[:, 'Size'] *= mess0['Direction']
#        ttypes[:-1, 0] = mess0['Time']
#        ttypes[-1, 0] = mess0.loc[i, 'Time']
#        ttypes[[np.arange(self.input_length - 1), mess0['Type']]] = 1
#        updates[[np.arange(self.input_length - 1), 
#                 np.clip(mess0['Price'], 0, 2*self.keep_lvl-1)]] = list(mess0['Size'] * mess0['Direction'])  
#        
#        if self.diffs:
#            future_lvls -= updates[-1:]
#        if return_xy:
#            return np.c_[ttypes, updates], future_lvls
#        return np.r_[np.c_[ttypes, updates],
#                     np.c_[np.zeros((1,6)), future_lvls]]
            
            
#
#def get_n_steps(end, n, book, mess, keeplvl=50):
#    assert n <= end, 'n=%d too large for end=%d' % (n, end) 
#    book1 = book.loc[end - n + 1:end].copy()
#    state = np.zeros((n, 2*keeplvl))
#    refP = (book1.loc[end, 'AskP 1'] + book1.loc[end, 'BidP 1'])//200 * 100
#    Pcols = [c for c in book1.columns if 'P' in c]
#    book1.loc[:, Pcols] = (book1[Pcols] - refP)/100 + keeplvl
#    for i in range(n):
#        for j in range(1, len(Pcols)//2 + 1):
#            lvl = book1.loc[end - n + 1 + i, 'BidP %d' % j]
#            assert lvl - int(lvl) == 0, 'wrong lvl: %f' % lvl
#            if lvl < 0:
#                break
#            state[i, int(lvl)] = -book1.loc[end - n + 1 + i, 'BidV %d' % j]
#        for j in range(1, len(Pcols)//2 + 1):
#            lvl = book1.loc[end - n + 1 + i, 'AskP %d' % j]
#            assert lvl - int(lvl) == 0, 'wrong lvl: %f' % lvl
#            if lvl >= 2 * keeplvl:
#                break
#            state[i, int(lvl)] = book1.loc[end - n + 1 + i, 'AskV %d' % j]   
#    
#    return state
#
#    
#    
#def get_events_and_state(end, n, book, mess, keeplvl=50, time=.01):
#    assert n <= end, 'n=%d too large for end=%d' % (n, end)
#    future = end - 1 + (mess['Time'][end:] <= mess['Time'][end] + time).sum()
#    current_state = book.loc[end].copy()
#    future_state = book.loc[future].copy()
#    updates = np.zeros((n, 2*keeplvl))
#    future_lvls = np.zeros((1, 2*keeplvl))
#    ttypes = np.zeros((n, 6))
#    refP = (current_state['AskP 1'] + current_state['BidP 1'])//200 * 100
#    for state, lvl_array in zip([current_state, future_state], [updates[-1], future_lvls[0]]):
#        Pcols = [c for c in state.index if 'P' in c]
#        state[Pcols] = (state[Pcols] - refP)/100 + keeplvl
#        for j in range(1, len(Pcols)//2 + 1):
#            lvl = state['BidP %d' % j]
#            assert lvl - int(lvl) == 0, 'wrong lvl: %f' % lvl
#            if lvl < 0:
#                break
#            lvl_array[int(lvl)] = -state['BidV %d' % j]
#        for j in range(1, len(Pcols)//2 + 1):
#            lvl = state['AskP %d' % j]
#            assert lvl - int(lvl) == 0, 'wrong lvl: %f' % lvl
#            if lvl >= 2 * keeplvl:
#                break
#            lvl_array[int(lvl)] = state['AskV %d' % j]   
#    mess0 = mess.loc[end - n + 2: end, :].copy()
#    mess0.loc[:, 'Price'] = np.int32((mess0['Price'] - refP)//100 + keeplvl)
#    mess0.loc[:, 'Size'] *= mess0['Direction']
#    ttypes[:-1, 0] = mess0['Time']
#    ttypes[-1, 0] = mess0.loc[end, 'Time']
#    ttypes[[np.arange(n-1), mess0['Type']]] = 1
#    updates[[np.arange(n-1), np.clip(mess0['Price'], 0, 2*keeplvl-1)]] = list(mess0['Size'] * mess0['Direction'])  
#    
#    return np.c_[ttypes, updates], future_lvls - updates[-1]
#
#    