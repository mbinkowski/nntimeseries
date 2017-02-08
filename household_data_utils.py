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
    
class HouseholdGenerator(Generator):
    def __init__(self, filename='household.pkl', 
                 url='https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip',
                 train_share=.8, input_length=1, output_length=1, verbose=1, 
                 limit=np.inf, batch_size=16):
        self.filename = filename
        self.url = url
        if self.filename not in os.listdir('data'):
            X = download_and_unzip(url=url, verbose=verbose, 
                                   filename=filename, limit=limit)
        else:
            X = pd.read_pickle('data/' + filename)
        nanno = np.isnan(X[X.columns[1:]]).sum(axis=1)
        self.no_of_nan_rows = (nanno > 0).sum()
        X = X.loc[nanno == 0]
        X.sort_values(by='datetime', inplace=True)
        super(HouseholdGenerator, self).__init__(X, train_share=train_share, 
                                                input_length=input_length, 
                                                output_length=output_length, 
                                                verbose=verbose, limit=limit,
                                                batch_size=batch_size,
                                                excluded=['datetime'])

    
