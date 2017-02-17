# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 13:57:01 2016

@author: mbinkowski
"""
from __init__ import *

def list_of_param_dicts(param_dict):
    vals = list(prod(*[v for k, v in param_dict.items()]))
    keys = list(prod(*[[k]*len(v) for k, v in param_dict.items()]))
    return [dict([(k, v) for k, v in zip(key, val)]) for key, val in zip(keys, vals)]
    
class ModelRunner(object):
    def __init__(self, param_dict, data_list, model, save_file, hdf5_dir='hdf5_keras_model_files'):
        self.param_list = list_of_param_dicts(param_dict)
        self.data_list = data_list
        self.model = model
        self.save_file = save_file
        self.cdata = None
        self.cp = None
        self.cresults = None
        self.time0 = time.time()
        self.hdf5_dir = hdf5_dir
        if hdf5_dir not in os.listdir(os.getcwd()):
            os.mkdir(hdf5_dir)
        
    def _read_results(self):
        if self.save_file.split('/')[1] in os.listdir(self.save_file.split('/')[0]):
        #    results = []
            results = [v for k, v in pd.read_pickle(self.save_file).T.to_dict().items()]
        else:
            results = []
        return results
    
    def _get_hdf5_name(self):
        n = 0
        files = os.listdir(self.hdf5_dir)
        if len(files) > 0:
            n = int(np.max([int(f.split('_')[0]) for f in files])) + 1
        t = datetime.datetime.now().isoformat().replace(':', '.')
        code = ''.join(np.random.choice([l for l in string.ascii_uppercase], 5))
        return '%s/%06d_%s_%s_RunMod.h5' % (self.hdf5_dir, n, t, code)
    
    def run(self, trials=3, log=False, read_file=None, limit=1, irrelevant=[]):
        if log:
            old_stdout = sys.stdout
            log_name = self.save_file.replace('results', 'logs')[:-4]   
            log_file = open(log_name + time.strftime("%x").replace('/', '.') + '.txt', 'w', buffering=1)
            sys.stdout = log_file
        self.cresults = self._read_results()
        unsuccessful_settings = []
        for params in self.param_list:
            for data in self.data_list:
                if limit < np.inf:
                    already_computed = self.lookup_setting(read_file=read_file,
                                                           params=params, data=data,
                                                           irrelevant=irrelevant)
                    if already_computed >= limit:
                        print('Found %d (>= limit = %d) computed results for the setting:' % (already_computed, limit))
                        print([data, params])
                        continue
                    else:
                        required_success = limit - already_computed
                        print('Found %d (< limit = %d) computed results for the setting.' % (already_computed, limit))
                else:
                    required_success = 1
                success, errors = 0, 0
                setting_time = time.time()
                while (errors < trials) and (success < required_success):
#                    try:
                    print(data + ' success: %d, errors: %d' % (success, errors))
                    print(params)
                    self.cdata = data
                    self.cp = params
                    history, nn, reducer = self.model(data, params)
                    self.nn = nn
                    self.reducer = reducer
                    self.history = history
                    model_results = history.history
                    model_results.update(params)
                    hdf5_name = self._get_hdf5_name()
                    print('setting time %.2f' % (time.time() - setting_time))
                    nn.save(hdf5_name)
                    model_results.update(
                        {'training_time': time.time() - setting_time,
                         'datetime': datetime.datetime.now().isoformat(),
                         'dt': datetime.datetime.now(),
                         'date': datetime.date.today().isoformat(),
                         'data': data,
                         'hdf5': hdf5_name,
                         'total_params': np.sum([np.sum([np.prod(K.eval(w).shape) for w in l.trainable_weights]) for l in nn.layers])
#                             'json': nn.to_json(),
#                             'model_params': reducer.saved_layers
                         }
                    )
                    self.cresults.append(model_results)
                    pd.DataFrame(self.cresults).to_pickle(self.save_file)
                    success += 1
#                    except Exception as e:
#                        errors += 1
#                        print(e)
                if success < required_success:
                    unsuccessful_settings.append([data, params])
        #    with open(save_file, 'wb') as f:
        #        pickle.dump(results, f)
        with open(self.save_file[:-4] + 'failed.pikle', 'wb') as f:
            pickle.dump(unsuccessful_settings, f)
        if log:
            sys.stdout = old_stdout
            log_file.close()
        return self.cresults
        
    def lookup_setting(self, read_file, params, data, irrelevant):
        if read_file is None:
            already_computed = self.cresults
        else:
            already_computed = [v for k, v in pd.read_pickle(read_file).T.to_dict().items()]
        count = 0
        for res in already_computed:
            if res['data'] != data:
                continue
            par_ok = 1
            for k, v in params.items():
                if k in irrelevant:
                    continue
                if k not in res:
                    par_ok = 0
                    break
                if res[k] != v:
                    par_ok = 0
                    break
            count += par_ok
        return count

class Generator(object):
    def __init__(self, X, train_share=(.8, 1), input_length=1, output_length=1, 
                 verbose=1, limit=np.inf, batch_size=16, excluded=[], 
                 diffs=False):
        self.X = X
        self.diffs = diffs
        if limit < np.inf:
            self.X = self.X.loc[:limit]
        self.train_share = train_share
        self.input_length = input_length
        self.output_length = output_length
        self.l = input_length + output_length
        self.verbose = verbose
        self.batch_size = batch_size
        self.n_train = int(((self.X.shape[0] - diffs) * train_share[0] - self.l)/batch_size) * batch_size + self.l
        self.n_all = self.n_train + int(((self.X.shape[0] - diffs) * train_share[1] - self.n_train - self.l)/batch_size) * batch_size + self.l
        self.excluded = excluded
        self.cols = [c for c in self.X.columns if c not in self.excluded]
        self._scale()
    
    def asarray(self, cols=None):
        if cols is None:
            cols = self.cols
        return np.asarray(self.X[cols], dtype=np.float32)
        
    def get_dim(self):
        return self.asarray().shape[1]

    def get_target_cols(self, ids=True, cols='default'):
        if cols in ['default', 'all']:
            if ids:
                return np.arange(len(self.cols))
            else:
                return self.cols
        elif type(cols) == list:
            if type(cols[0]) == str:
                return [(i if ids else c) for i, c in enumerate(self.cols) if c in cols]
            elif type(cols[0]) in [int, float]:
                return [(int(i) if ids else self.cols[int(i)]) for i in cols]
            else:
                raise Exception('cols = ' + repr(cols) + ' not supported')
        else:
            raise Exception('cols = ' + repr(cols) + ' not supported')

        
    def exclude_columns(self, cols):
        self.excluded += cols
        
    def _scale(self, exclude=None, exclude_diff=None):
        if exclude is None:
            exclude = self.excluded
        cols = [c for c in self.X.columns if c not in exclude]
        if self.diffs:
            diff_cols = [c for c in self.X.columns if c not in exclude_diff]
            self.X.loc[:, diff_cols] = self.X.loc[:, diff_cols].diff()
            self.X = self.X.loc[self.X.index[1:]]
        self.means = self.X.loc[:self.n_train, cols].mean(axis=0)
        self.stds = self.X.loc[:self.n_train, cols].std(axis=0)
        self.X.loc[:, cols] = (self.X[cols] - self.means)/(self.stds + (self.stds == 0)*.001)
        
    def gen(self, mode='train', batch_size=None, func=None, shuffle=True, 
            n_start=0, n_end=np.inf):
        if batch_size is None:
            batch_size = self.batch_size
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
        if not shuffle:
            if (n_end - n_start - self.l) % batch_size != 0:
                raise Exception('For non-shuffled input (for RNN) batch_size must divide n_end - n_start - self.l')
            if mode == 'valid':
                n_start -= self.l - 1
                n_end -= self.l - 1
        XX = self.asarray()
        x = []
        while True:
            order = np.arange(n_start + self.input_length, n_end - self.output_length)
            if shuffle:
                order = np.random.permutation(order)
            else:
                order = order.reshape(batch_size, len(order)//batch_size).transpose().ravel()
            for i in order:
                if len(x) == batch_size:
                    yield func(np.array(x))
                    x = []
                x.append(XX[i - self.input_length: i + self.output_length, :])
    
    def make_io_func(self, io_form, cols=[0], input_cols=None):
        cols = self.get_target_cols(cols)
        il = self.input_length
        if io_form == 'regression':
            def regr(x):
                osh = (x.shape[0], (x.shape[1] - il) * len(cols))
                return (x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols], 
                        x[:, il:, cols].reshape(osh))
            return regr
        
        elif io_form == 'flat_regression':
            def regr(x):
                ish = (x.shape[0], il * x.shape[2])
                osh = (x.shape[0], (x.shape[1] - il) * len(cols))
                inp =  x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols]
                return (inp.reshape(ish), 
                        x[:, il:, cols].reshape(osh))
            return regr
    
        elif io_form == 'vi_regression':
            def regr(x):
        #        osh = (x.shape[0], (x.shape[1] - il) * len(cols))
                return ({'inp': x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols], 
                         'value_input': x[:, :il, cols]},
                        x[:, il:, cols])
            return regr
        
        elif io_form == 'cvi_regression':
            def regr(x):
        #        osh = (x.shape[0], (x.shape[1] - il) * len(cols))
                return (
                    {'inp': x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols], 
                     'value_input': x[:, :il, cols]},
                    {'main_output': x[:, il:, cols],
                     'value_output': np.concatenate(il*[x[:, il: il+1, cols]], axis=1)}
                )           
            return regr
        else:
            raise Exception('io_form' + repr(io_form) + 'not implemented')
        
def make_regression(input_length=60, cols=[0], input_cols=None):
    def regr(x):
        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        return (x[:, :input_length, :] if (input_cols is None) else x[:, :input_length, input_cols], 
                x[:, input_length:, cols].reshape(osh))
    return regr
    
def make_flat_regression(input_length=60, cols=[0], input_cols=None):
    def regr(x):
        ish = (x.shape[0], input_length * x.shape[2])
        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        inp =  x[:, :input_length, :] if (input_cols is None) else x[:, :input_length, input_cols]
        return (inp.reshape(ish), 
                x[:, input_length:, cols].reshape(osh))
    return regr

def make_vi_regression(input_length=60, cols=[0], input_cols=None):
    def regr(x):
#        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        return ({'inp': x[:, :input_length, :] if (input_cols is None) else x[:, :input_length, input_cols], 
                 'value_input': x[:, :input_length, cols]},
                x[:, input_length:, cols])
    return regr
    
def make_cvi_regression(input_length=60, cols=[0], input_cols=None):
    def regr(x):
#        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        il = input_length
        return (
            {'inp': x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols], 
             'value_input': x[:, :il, cols]},
            {'main_output': x[:, il:, cols],
             'value_output': np.concatenate(il*[x[:, il: il+1, cols]], axis=1)}
        )           
    return regr