# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:42:53 2017

@author: mbinkowski
"""
import os
os.chdir('C://Users//mbinkowski//cdsol-r-d.cluster//cdsol-r-d.machine_learning_studies//nntimeseries/')
from __init__ import *
import utils

DATASET = 'data/artificialET1SS0n100000S16.csv'
ALL_RESULTS_FILE = 'cluster_results.pkl'
LIMIT = 5
#from artificial_data_utils import ArtificialGenerator as generator

log = False

key_dict = {
    'cvi2': ['verbose', 'train_share', 'input_length', 'output_length', 'patience', 'filters', 'act', 'dropout', 'kernelsize', 
             'layers_no', 'poolsize', 'architecture', 'batch_size', 'objective', 'norm', 'nonnegative', 'connection_freq', 
             'aux_weight', 'shared_final_weights', 'resnet', 'diffs', 'target_cols'],
    'cnn': ['verbose', 'train_share', 'input_length', 'output_length', 'patience', 'filters', 'act', 'dropou', 'kernelsize',
            'poolsize', 'layers_no', 'batch_size', 'objective', 'norm', 'maxpooling', 'resnet', 'diffs', 'target_cols'],
    'lstm': ['verbose', 'train_share', 'input_length', 'output_length', 'patience', 'layer_size', 'act', 'dropout',
             'layers_no', 'batch_siz', 'objective', 'norm', 'diffs', 'target_cols']
}

function_dict = {
    'cvi2' : CVI2model,
    'cnn': CNNmodel,
    'lstm': LSTMmodel
}

df = pd.read_pickle(WDIR + '/results/' + ALL_RESULTS_FILE)

df = df[df['data'] == DATASET]

key = 'layers_no'
df['str' + key] = df[key].apply(str)
dfs = []
for model, gbkey in zip(['cvi2', 'cnn', 'lstm'], ['str' + key, 'filters', 'layer_size']):
    df0 = df[df['model'] == model]
    df0 = df0[df0.groupby([gbkey, 'data'])['best_mse'].transform(min) == df0['best_mse']].copy()
    df0['specialkey'] = df0[gbkey].apply(lambda x: model + ' ' + str(x))
    dfs.append(df0)
df = pd.concat(dfs)

df = df[~df['specialkey'].isin(['lstm 8.0', "cvi2 {'sigs': 1, 'offs': 1}"])]

settings = []

def safely2int(k):
    if (type(k) == float) or (type(k) == np.float64) or (type(k) == np.float32):
        if k % 1 == 0:
            return int(k)
    return k

for idx in df.index:
    model = df.loc[idx, 'model']
    model_function = function_dict[model]
    param_dict = dict([(k, [safely2int(v)]) for k, v in df.loc[idx].to_dict().items() if k in key_dict[model]])
    dataset = [df.loc[idx, 'data']]
    save_file = dataset[0].split('/')[1][:-4] + '_best_model_test.pkl'
    settings.append((model_function, param_dict, dataset, save_file))

for limit in np.arange(2, LIMIT + 1):    
    for setting in list(np.random.permutation(settings)):
        model_function, param_dict, dataset, save_file = setting
        runner = utils.ModelRunner(param_dict, dataset, model_function, save_file)
        if save_file in os.listdir(WDIR + '/results'):
            runner.run(log=log, limit=limit - 1, read_file='/results/' + save_file)
        else:
            runner.run(log=log, limit=limit, read_file='/results/' + ALL_RESULTS_FILE)

