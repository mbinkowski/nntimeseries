"""
Created on Thu Dec 15 13:57:01 2016

@author: mbinkowski

This file provides utilities for downloading and generating training and 
validation samples from household electricity datasetc power consumption dataset

https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
"""

from ._imports_ import *
from .config import WDIR

def download_and_unzip(url='https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
                       verbose=1, filename='household', limit=np.inf):
    import urllib, zipfile
    if 'data' not in os.listdir(WDIR):
        os.mkdir('data')
    t0 = time.time()
    if verbose > 0:
        print('Downloading data from ' + url + '...')
    tmp = 'tmp%d' % int(np.random.rand(1)*1000000)
    u = urllib.request.urlretrieve(url, os.path.join(WDIR, 'data', tmp + '.zip'))
    if verbose > 0:
        print('time = %.2fs, data downloaded. Extracting archive...' % (time.time() - t0))
    zip_ref = zipfile.ZipFile(u[0], 'r')
    zip_ref.extractall(os.path.join(WDIR, 'data', tmp))
    zip_ref.close()
    os.remove(os.path.join(WDIR, 'data', tmp + '.zip'))
    if verbose > 0:
        print('time = %.2fs, data extracted. Parsing text file...' % (time.time() - t0))
    X = pd.read_csv(os.path.join(WDIR, 'data', tmp, 'household_power_consumption.txt'), sep=';', 
                    parse_dates={'datetime': [0,1]}, dtype=np.float32, 
                    na_values=['?'], nrows=limit if (limit < np.inf) else None)
    X['time'] = X['datetime'].apply(lambda x: x.hour*60 + x.minute)
#    X['time'] = X['datetime'].apply(lambda x: (x - pd.Timestamp(x.date())))/np.timedelta64(1, 's')
    filepath = os.path.join(WDIR, 'data', filename + '.pkl')
    X.to_pickle(filepath)
    if verbose > 0:
        print("time = %.2fs, data converted and saved as '%s'" % (time.time() - t0, filepath))
    os.remove(os.path.join(WDIR, 'data', tmp, 'household_power_consumption.txt'))
    os.rmdir(os.path.join(WDIR, 'data', tmp))
    return X
    
class HouseholdGenerator(utils.Generator):
    """
    Class that provides sample generator for Household Electricity Dataset. 
    
    """
    def __init__(self, filename='household', 
                 url='https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
                 train_share=(.8, 1.), input_length=1, output_length=1, verbose=1, 
                 limit=np.inf, batch_size=16, diffs=False):
        self.filename = filename
        self.url = url
        self.verbose = verbose
        self.limit = limit
        X = self.get_X()
        super(HouseholdGenerator, self).__init__(X, train_share=train_share, 
                                                input_length=input_length, 
                                                output_length=output_length, 
                                                verbose=verbose, limit=limit,
                                                batch_size=batch_size,
                                                excluded=['datetime'],
                                                diffs=diffs)

    def get_X(self):
        if self.filename not in os.listdir(os.path.join(WDIR, 'data')):
            X = download_and_unzip(url=self.url, verbose=self.verbose, 
                                   filename=self.filename, limit=self.limit)
        else:
            X = pd.read_pickle(os.path.join(WDIR, 'data', self.filename))
        nanno = np.isnan(X[X.columns[1:]]).sum(axis=1)
        self.no_of_nan_rows = (nanno > 0).sum()
        X = X.loc[nanno == 0]
        X.sort_values(by='datetime', inplace=True)
        return X
        
    def get_target_col_ids(self, ids=True, cols='default'):
        if cols in ['all', 'default']:
            cols = self.cols 
        assert hasattr(cols, '__iter__')
        return [(i if ids else c) for i, c in enumerate(self.cols) if ('time' not in c) and (c in cols)]
    

class HouseholdAsynchronousGenerator(HouseholdGenerator):
    def __init__(self, filename='household_async.pkl', 
                 url='https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
                 train_share=(.8, 1.), input_length=1, output_length=1, verbose=1, 
                 limit=np.inf, batch_size=16, diffs=False, new_schedule=False,
                 duration_type='deterministic'):
        self.filename = filename
        self.url = url
        self.verbose = verbose
        self.limit = limit
        X = self.get_X()
        
        self.value_cols = [c for c in X.columns if 'time' not in c]
        self.ind_cols = [c +'_ind' for c in self.value_cols]
        self.schedule_file = self.filename + '_schedule.pkl'
        if (self.schedule_file in os.listdir(os.path.join(WDIR, 'data'))) and (not new_schedule):
            schedule = pd.read_pickle(os.path.join(WDIR, 'data', self.schedule_file))
        else:
            schedule = self.generate_schedule(duration_type=duration_type)
            schedule.to_pickle(os.path.join(WDIR, 'data', self.schedule_file))
        X['value'] = 0
        for c in self.value_cols:
            X['value'] += X[c] * schedule[c + '_ind']
        X = pd.concat([X, schedule], axis=1)
        X = X.loc[X[self.ind_cols].sum(axis=1) > 0, :]
        
        super(HouseholdGenerator, self).__init__(X, train_share=train_share, 
                                                input_length=input_length, 
                                                output_length=output_length, 
                                                verbose=verbose, limit=limit,
                                                batch_size=batch_size,
                                                excluded=['datetime'],
                                                diffs=diffs)        
        
        self.cols = self.value_cols + self.ind_cols + ['time', 'value']

           
    def generate_schedule(self, duration_type='deterministic'):
        N, d = self.X.shape
        frequencies = np.random.permutation(1.5**np.arange(d - 2))
        frequencies /= frequencies.sum()
        schedule = np.random.multinomial(1, pvals=frequencies, size=(N,))
        if duration_type == 'deterministic':
            v = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1]
            valid = np.array((v * (N // len(v) + 1))[:N])
        elif duration_type == 'random':
            valid = [1]
            while len(valid) < N:
                duration = int(np.random.exponential(2.))
                valid += [0] * duration + [1]
            valid = np.array(valid[:N])
        schedule *= valid.reshape(N, 1)
        return pd.DataFrame(schedule, columns=[c +'_ind' for c in self.value_cols])
        
    def make_io_func(self, io_form, cols='default', input_cols=None):
        if input_cols is None:
            input_cols = ['value', 'time'] + self.ind_cols
            input_cols = [i for i, c in enumerate(self.cols) if c in input_cols]
        return super(HouseholdAsynchronousGenerator, self).make_io_func(
            io_form=io_form, cols=cols, input_cols=input_cols
        )
        
    def get_target_col_ids(self, ids=True, cols='default'):
        return super(HouseholdAsynchronousGenerator, self).get_target_col_ids(
            ids=ids, cols=self.value_cols if (cols in ['all', 'default']) else cols
        )
        