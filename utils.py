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
    
    def run(self, trials=3, log=False):
        if log:
            old_stdout = sys.stdout
            log_name = self.save_file.replace('results', 'logs')[:-4]   
            log_file = open(log_name + time.strftime("%x").replace('/', '.') + '.txt', 'w', buffering=1)
            sys.stdout = log_file
        self.cresults = self._read_results()
        unsuccessful_settings = []
        for params in self.param_list:
            for data in self.data_list:
                success = -trials
                setting_time = time.time()
                while success < 0:
#                    try:
                    print(data + ' success: ' + repr(success))
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
                         'hdf5': hdf5_name
#                             'json': nn.to_json(),
#                             'model_params': reducer.saved_layers
                         }
                    )
                    self.cresults.append(model_results)
                    pd.DataFrame(self.cresults).to_pickle(self.save_file)
                    success = 1
#                    except Exception as e:
#                        success += 1
#                        print(e)
                if success < 1:
                    unsuccessful_settings.append([data, params])
        #    with open(save_file, 'wb') as f:
        #        pickle.dump(results, f)
        with open(self.save_file[:-4] + 'failed.pikle', 'wb') as f:
            pickle.dump(unsuccessful_settings, f)
        if log:
            sys.stdout = old_stdout
            log_file.close()
        return self.cresults


def make_regression(input_length=60, cols=[0]):
    def regr(x):
        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        return (x[:, :input_length, :], 
                x[:, input_length:, cols].reshape(osh))
    return regr
    
def make_flat_regression(input_length=60, cols=[0]):
    def regr(x):
        ish = (x.shape[0], input_length * x.shape[2])
        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        return (x[:, :input_length, :].reshape(ish), 
                x[:, input_length:, cols].reshape(osh))
    return regr

def make_vi_regression(input_length=60, cols=[0]):
    def regr(x):
#        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        return ({'inp': x[:, :input_length, :], 
                 'value_input': x[:, :input_length, cols]},
                x[:, input_length:, cols])
    return regr
    
def make_cvi_regression(input_length=60, cols=[0]):
    def regr(x):
#        osh = (x.shape[0], (x.shape[1] - input_length) * len(cols))
        il = input_length
        return (
            {'inp': x[:, :il, :], 
             'value_input': x[:, :il, cols]},
            {'main_output': x[:, il:, cols],
             'value_output': np.concatenate(il*[x[:, il: il+1, cols]], axis=1)}
        )           
    return regr