# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:57:01 2016

@author: mbinkowski
"""
from __init__ import *

def download_and_unzip(url='https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
                       verbose=1, filename='household.pkl', limit=np.inf):
    import urllib, zipfile
    if 'data' not in os.listdir(os.getcwd()):
        os.mkdir('data')
    t0 = time.time()
    if verbose > 0:
        print('Downloading data from ' + url + '...')
    tmp = 'tmp%d' % int(np.random.rand(1)*1000000)
    u = urllib.request.urlretrieve(url, 'data/' + tmp + '.zip')
    if verbose > 0:
        print('time = %.2fs, data downloaded. Extracting archive...' % (time.time() - t0))
    zip_ref = zipfile.ZipFile(u[0], 'r')
    zip_ref.extractall('data/' + tmp)
    zip_ref.close()
    os.remove('data/' + tmp + '.zip')
    if verbose > 0:
        print('time = %.2fs, data extracted. Parsing text file...' % (time.time() - t0))
    X = pd.read_csv('data/' + tmp + '/household_power_consumption.txt', sep=';', 
                    parse_dates={'datetime': [0,1]}, dtype=np.float32, 
                    na_values=['?'], nrows=limit if (limit < np.inf) else None)
    X['time'] = X['datetime'].apply(lambda x: x.hour*60 + x.minute)
#    X['time'] = X['datetime'].apply(lambda x: (x - pd.Timestamp(x.date())))/np.timedelta64(1, 's')
    filepath = 'data/' + filename
    X.to_pickle(filepath)
    if verbose > 0:
        print("time = %.2fs, data converted and saved as '%s'" % (time.time() - t0, filepath))
    os.remove('data/' + tmp + '/household_power_consumption.txt')
    os.rmdir('data/' + tmp)
    return X
    
class Generator(object):
    def __init__(self, filename='household.pkl', 
                 url='https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
                 train_share=.8, input_length=1, output_length=1, verbose=1, 
                 limit=np.inf):
        self.filename = filename
        self.url = url
        self.train_share = train_share
        self.input_length = input_length
        self.output_length = output_length
        self.verbose = verbose
        if self.filename not in os.listdir('data'):
            self.X = download_and_unzip(url=url, verbose=verbose, 
                                        filename=filename, limit=limit)
        else:
            self.X = pd.read_pickle('data/' + filename)
            if limit < np.inf:
                self.X.loc[:limit]
        self.cnames = self.X.columns
        nanno = np.isnan(self.X[self.cnames[1:]]).sum(axis=1)
        self.no_of_nan_rows = (nanno > 0).sum()
        self.X = self.X.loc[nanno == 0]
        self.n_train = int(self.X.shape[0] * train_share)
        self.n_all = self.X.shape[0]
        self.X.sort_values(by='datetime', inplace=True)
        self._scale()
    
    def asarray(self, cols=None):
        if cols is None:
            cols = self.cnames[1:]
        return np.asarray(self.X[cols], dtype=np.float32)
        
    def _scale(self, exclude=['datetime']):
        cols = [c for c in self.X.columns if c not in exclude]
        self.means = self.X.loc[:self.n_train, cols].mean(axis=0)
        self.stds = self.X.loc[:self.n_train, cols].std(axis=0)
        self.X.loc[:, cols] = (self.X[cols] - self.means)/self.stds
        
    def gen(self, mode='train', batch_size=16, func=None, shuffle=True, 
            n_start=0, n_end=np.inf):
        if func is None:
            func = lambda x: (x[:, :self.input_length, :], x[:, self.input_length:, :])
        if mode=='train':
            n_end = self.n_train
        elif mode == 'valid':
            n_start = self.n_train
            n_end = self.n_all
        elif mode == 'manual':
            assert n_end < self.n_all
            assert n_start >= 0
            assert n_end > n_start
        else:
            raise Exception('invalid mode')
        XX = self.asarray()
        x = []
        while True:
            order = np.arange(n_start + self.input_length, n_end - self.output_length)
            if shuffle:
                order = np.random.permutation(order)
            for i in order:
                if len(x) == batch_size:
                    yield func(np.array(x))
                    x = []
                x.append(XX[i - self.input_length: i + self.output_length, :])