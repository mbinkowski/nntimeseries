"""
Utilities file.

The file contains i.a. the ModelRunner and Generator classes.
"""
from ._imports_ import *
from .config import WDIR, SEP
from . import keras_utils, user
import sys

def list_of_param_dicts(param_dict):
    """
    Function to convert dictionary of lists to list of dictionaries of
    all combinations of listed variables. Example
    list_of_param_dicts({'a': [1, 2], 'b': [3, 4]})
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    vals = list(prod(*[v for k, v in param_dict.items()]))
    keys = list(prod(*[[k]*len(v) for k, v in param_dict.items()]))
    return [dict([(k, v) for k, v in zip(key, val)]) for key, val in zip(keys, vals)]

        
def get_param_no(nn):
    return int(np.sum([np.sum([np.prod(K.eval(w).shape) for w in l.trainable_weights]) for l in nn.layers]))
    
    
class ModelRunner(object):
    """
    Class that defines a grid-search task with specified hyperparameters and 
    datasets.
    Initialization arguments:
        param_dict  - dictionary of the form {param_name: param_list, ...}
                      where 'param_list' is a list of values for parameter 
                      'param_name' to search over
        data_list   - list of datasets to test. Each dataset should have a form
                      of valid input to the model input function                    
        model       - function defining model to evaluate; should take 2 
                      arguments, first of which has to be a dictionary of the 
                      form {param_name, param_value} and second of arbitrary form
        save_file   - path to the file to save_results in 
        hdf5_dir    - directory to save trained models using keras Model.save()
                      method
    """    
    def __init__(self, param_dict, data_list, save_file, 
                 hdf5_dir='hdf5_keras_model_files'):
        self.param_list = list_of_param_dicts(param_dict)
        self.data_list = data_list
        self.save_file = os.path.join(WDIR, save_file)
        self.cdata = None
        self.cp = None
        self.cresults = None
        self.time0 = time.time()
        self.hdf5_dir = os.path.join(WDIR, hdf5_dir)
        if hdf5_dir not in os.listdir(WDIR):
            os.mkdir(self.hdf5_dir)
        
    def _read_results(self):
        if self.save_file.split(SEP)[-1] in os.listdir(SEP.join(self.save_file.split(SEP)[:-1])):
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
        return '%s%s%06d_%s_%s_RunMod.h5' % (self.hdf5_dir, SEP, n, t, code)
    
    def run(self, model_class, trials=3, log=False, read_file=None, limit=1, 
            irrelevant=[]):
        """
        Function that launches grid search, saves and returns results.
        Arguments:
            trials      - (int) number of trials to succesfully run single 
                          model setting; when number of errors exceeds 'trials'
                          grid serach goes to the next hyperparameter setting
            log         - if True, stdout is passed to the log file saved in
                          logs directory
            read_file   - file to read the previously computed results from.
                          If a setting has already been tested enough many 
                          times, grid serach passes to the next setting
            limit       - required number of successful runs for each single
                          parameter setting
            irrelevant  - list of paramters irrelevant while comparing a 
                          setting with the previously computed results. This
                          parameter has no impact if read file is not specified
        Returns
            list of dictionaries; each dictionary contains data from keras 
            History.history dictionary, parameter dictionary and other data
        """
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
                        print(data)
                        for k, v in params.items():
                            print(str(k).rjust(15) + ': ' +  str(v))
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
                    print(data)
                    print('As yet, for this configuration: success: %d, errors: %d' % (success, errors))
                    for k, v in params.items():
                        print(str(k).rjust(15) + ': ' +  str(v))
                    self.cdata = data
                    self.cp = params
                    print("using " + repr(model_class) + " to build the model")
                    model = model_class(data, params, os.path.join(WDIR, 'tensorboard'))
#                    history, nn, reducer = model.run()
                    model_results, nn = model.run()
                    self.nn = nn
#                    self.reducer = reducer
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
        """
        Function that counts already computed results .
        Arguments:
            read_file   - file to read the previously computed results from
            params      - a dictionary of parameters to look for
            data        - dataset for which to look for
            irrelevant  - list of paramters irrelevant while comparing  
                          'params' with the previously computed results
        Returns
            number of times the given (parameter, data) setting occurs in
            training data saved in read_file
        """
        if read_file is None:
            already_computed = self.cresults
        else:
            already_computed = [v for k, v in pd.read_pickle(os.path.join(WDIR, read_file)).T.to_dict().items()]
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


class Model(object):
    """
    Abstract class defines the general model structure to be passed to 
    <utils.ModelRunner>.
    Classes that inherit from <nnts.utils.Model class> should implement 
    <build> method. 
    """
    def __init__(self, datasource, params, tensorboard_dir="." + SEP, 
                 tb_val_limit=1024):
        """
        Aruments:
            datasource      - correct argument to the generator object construtor
            params          - the dictionary with all of the model hyperparameters
            tensorboard_dir - directory to store TensorBoard logs
            tb_val_limit    - max number of validation samples to use by TensorBoard
        """
        self.name = "Model"
        self._set_params(params)
        self.datasource = datasource
        self.tensorboard_dir = tensorboard_dir
        self.tb_val_limit = tb_val_limit
        try:
            generator_class = get_generator(datasource)
        except:
            generator_class = UserGenerator
        print("Using " + repr(generator_class))
            
        self.G = generator_class(datasource, **params)
        self.idim, self.odim = self.G.get_dims(cols=self.target_cols)   
        self.nn, self.io_func, self.callbacks = self.build()
        
    def _set_params(self, params):
        self.train_share = (.8, 1)      # default delimeters of the training and validation shares
        self.input_length = 60          # default input length 
        self.output_length = 1          # default no. of timesteps to predict (only 1 impelemented)
        self.verbose = 1                # default verbosity
        self.batch_size = 128           # default batch size
        self.diffs = False              # if yes, work on 1st difference of series instead of original
        self.target_cols = 'default'    # 'default' or list of names of columns to predict
        self.patience = 5               # default no. of epoch after which learning rate will decrease if no improvement
        self.reduce_nb = 2              # defualt no. of learning rate reductions
        self.shuffle = True             # default wheather to shuffle batches during training
        self.__dict__.update(params)

    def build(self):
        """
        Classes that inherit from <nnts.utils.Model class> should implement 
        <build> method so that it returns:
            nn                 - keras.models.Model object
            io_func            - function that converts raw array to the input 
                                 form that feeds the model. Can be obtained through
                                 <nnts.utils.Generator.make_io_func> method
            callbacks          - list of keras.callbacks.Callback objects
        """   
        raise NotImplementedError("Called from an abstract class. Implement \
                                  <build> method in a derived class.")
    
    def run(self):
        """
        Returns:
            keras.callbacks.History object,
            kera.models.Model object,
            nnts.keras_utils.LrReducer object.
        """
        print('Total model parameters: %d' % get_param_no(self.nn))
        
        tb_dir = os.path.join(self.tensorboard_dir, 
                              datetime.date.today().isoformat(), self.name)
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        tensorboard = keras_utils.TensorBoard(
            log_dir=tb_dir, histogram_freq=1, write_images=True
        )
        if self.G.test:
            test_cb = keras_utils.Test(self.G, self.io_func, self.verbose)
            self.callbacks.append(test_cb)
        
        validation_size = self.G.n_valid - self.G.n_train - self.G.l
        self.tb_gen = self.G.gen('valid', func=self.io_func, shuffle=self.shuffle,
                            batch_size=min(validation_size, self.tb_val_limit))
        print()
        hist = self.nn.fit_generator(
            self.G.gen('train', func=self.io_func, shuffle=self.shuffle),
            steps_per_epoch = (self.G.n_train - self.G.l) // self.batch_size,
            epochs=1000,
            callbacks=self.callbacks + [tensorboard],
            validation_data=self.G.gen('valid', func=self.io_func, shuffle=self.shuffle),
            validation_steps=validation_size // self.batch_size,
            verbose=self.verbose
        )
        history = hist.history
        if self.G.test:
            history.update(test_cb.test_hist)
        return history, self.nn#, reducer        
        
        
class Generator(object):
    """
    Class that defines a generator that produces samples for fit_generator
    method of the keras Model class.
    Initialization arguments:
        X               - (pandas.DataFrame) data table
        train_share     - tuple of two numbers in range (0, 1) that provide % limits 
                      for training and validation samples
        input_length    - no. of timesteps in the input
        output_length   - no. of timesteps in the output
        verbose         - level of verbosity (corresponds to keras use of 
                          verbose argument)
        limit           - maximum number of timesteps-rows in the 'X' DataFrame
        batch_size      - batch size
        excluded        - columns from X to exclude
        diffs           - if True, X is replaced with table of 1st differences
                          of the input series
    """
    def __init__(self, X, train_share=(.8, 1), input_length=1, output_length=1, 
                 verbose=1, limit=np.inf, batch_size=16, excluded=[], 
                 diffs=False, exclude_diff=[], **kwargs):
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
        self.n_valid = self.n_train + int(((self.X.shape[0] - diffs) * train_share[1] - self.n_train - self.l)/batch_size) * batch_size + self.l
        if len(train_share) > 2:
            self.n_test = self.n_valid + int(((self.X.shape[0] - diffs) * train_share[2] - self.n_valid - self.l)/batch_size) * batch_size + self.l
            self.test = True
        else:
            self.test = False
        self.excluded = excluded
        self.cols = [c for c in self.X.columns if c not in self.excluded]
        self._scale(exclude_diff=exclude_diff)
    
    def asarray(self, cols=None):
        if cols is None:
            cols = self.cols
        return np.asarray(self.X[cols], dtype=np.float32)

    def get_target_col_ids(self, cols, ids=True):
        if cols in ['default', 'all']:
            if ids:
                return np.arange(len(self.cols))
            else:
                return self.cols
        elif hasattr(cols, '__iter__'):
            if type(cols[0]) == str:
                return [(i if ids else c) for i, c in enumerate(self.cols) if c in cols]
            elif type(cols[0]) in [int, float]:
                return [(int(i) if ids else self.cols[int(i)]) for i in cols]
            else:
                raise Exception('cols = ' + repr(cols) + ' not supported')
        else:
            raise Exception('cols = ' + repr(cols) + ' not supported')

    def get_dim(self):
        return self.asarray().shape[1]

    def get_dims(self, cols):
        return self.get_dim(), len(self.get_target_col_ids(cols=cols))
        
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

    def _get_ith_sample(self, i):
        return self.asarray()[i - self.input_length: i + self.output_length, :]
        
    def gen(self, mode='train', batch_size=None, func=None, shuffle=True, 
            n_start=0, n_end=np.inf):
        """
        Function that yields possibly infinitely many training/validation 
        samples.
        Arguments:
            mode        - if 'train' or 'valid', the first and last indices of 
                          returned timesteps are within boundaries defined by 
                          train_share at initilization; if 'manual' n_start and
                          n_end have to be provided
           batch_size   - if None, self.batch_size is used
           func         - function that is applied to each output sample;
                           can provide formatting or discarding certain dimentions
                          default: 
                              lambda x: (x[:, :self.input_length, :], 
                                         x[:, self.input_length:, :])
            shuffle     - wheather or not to shuffle samples every training epoch
            n_start, n_end - lower and upper limits of timesteps to appear in 
                             the generated samples. Irrelevant if mode != 'manual
        Yields
            sequence of samples func(x) where x is a numpy.array of consecutive 
            rows of X
                          
        """
        np.random.seed(123)
        if batch_size is None:
            batch_size = self.batch_size
        if func is None:
            func = lambda x: (x[:, :self.input_length, :], x[:, self.input_length:, :])
        if mode in ['train', 'valid']:
            order = np.arange(
                self.input_length, self.n_valid - self.output_length
            )
            o_len = int(len(order) * self.train_share[0] / self.train_share[1])
            order = order[:o_len] if (mode == 'train') else order[o_len:]
        elif mode == 'test':
            assert self.test, "Test sample undefinded. To enable 'test' mode define three delimiters for train_share."
            order = np.arange(self.n_valid + self.input_length, 
                              self.n_test - self.output_length)
        elif mode == 'manual':
            assert n_end < self.n_valid
            assert n_start >= 0
            assert n_end > n_start
            order = np.arange(n_start + self.input_length, n_end - self.output_length)
        else:
            raise Exception('invalid mode')
        if not shuffle:
            if (n_end - n_start - self.l) % batch_size != 0:
                raise Exception('For non-shuffled input (for RNN) batch_size must divide n_end - n_start - self.l')
            if mode == 'valid':
                n_start -= self.l - 1
                n_end -= self.l - 1
            order = np.arange(n_start + self.input_length, n_end - self.output_length)
        x = []
        while True:
            if shuffle:
                order = np.random.permutation(order)
            else:
                order = order.reshape(batch_size, len(order)//batch_size).transpose().ravel()
            for i in order:
                if len(x) == batch_size:
                    yield func(np.array(x))
                    x = []
                x.append(self._get_ith_sample(i))
    
    def make_io_func(self, io_form, cols, input_cols=None):
        """
        Function that defines input/output format function to be passed to 
        self.gen.
        Arguments:
            io_form     - string indicating input/output format
                'flat_regression': returns pair of 2d np.arrays (no time 
                                   dimension, only batch x sample_size)
                                   appropriate for Linear Regression
                'regression':      returns tuple (3d np.array, 2d np.array)
                                   first array formatted for LSTM and CNN nets
                'vi_regression':   format for SOCNN network wihtout auxiliary
                                   output
                'cvi_regression':  format for SOCNN network with auxiliary
                                   output   
            cols        - list of output column names (as of self.X DataFrame)
                          if 'default' all columns are passed
            input_cols  - list of input columns indices (as of self.X DataFrame)
                          if None all columns are passed
        Returns
            function that takes a numpy array as an input and returns 
            appropriately formatted input (usually as required by the keras 
            model)
                          
        """
        cols = self.get_target_col_ids(cols=cols)
        il = self.input_length
        if io_form == 'stateful_lstm_regression':
            def regr(x):
#                osh = (x.shape[0], il * len(cols))
                return (x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols], 
                        x[:, 1:, cols])
            return regr
        elif io_form == 'regression':
            def regr(x):
                osh = (x.shape[0], (x.shape[1] - il) * len(cols))
                return (x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols], 
                        x[:, il:, cols].reshape(osh))
            return regr            
        
        elif io_form == 'flat_regression':
            def regr(x):
                if input_cols is None:
                    ish = (x.shape[0], il * x.shape[2])
                    inp =  x[:, :il, :]
                else:
                    ish = (x.shape[0], il * len(input_cols))
                    inp = x[:, :il, input_cols]
                osh = (x.shape[0], (x.shape[1] - il) * len(cols))              
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
            
        elif io_form == 'strategy1':
            def regr(x):
                ab = x[:, [il, -1], cols]
                p = np.array([ab[:, 0, 0] > ab[:, 1, 1], 
                              (ab[:, 0, 0] <= ab[:, 1, 1]) & (ab[:, 0, 1] >= ab[:, 1, 0]), 
                              ab[:, 0, 1] < ab[:, 1, 0]]).transpose()
                return (
                    {'inp': x[:, :il, :] if (input_cols is None) else x[:, :il, input_cols], 
                     'value_input': x[:, :il, cols]},
                    {'value_output': ab,
                     'prob_output': p,
                     'full_output': np.concatenate(p, ab.reshape(self.batch_size, 4))}
                )
            return regr            
        else:
            raise Exception('io_form' + repr(io_form) + 'not implemented')


class UserGenerator(Generator):
    """
    Class that defines a user-friendly generator that produces samples for 
    fit_generator method of the keras Model class.
    Initialization arguments:
        data                  - path to pickled <pandas.DataFrame>
        input_column_names    - list of column names to use as input
        target_column_names   - list of names of columns to predict 
        diff_column_names     - list of columns to take first difference of 
                                at preprocessing stage
        train_share     - tuple of two numbers in range (0, 1) that provide % limits 
                          for training and validation samples
        input_length    - no. of timesteps in the input
        output_length   - no. of timesteps in the output
        batch_size      - batch size
        verbose         - level of verbosity (corresponds to keras use of 
                          verbose argument)
        limit           - maximum number of timesteps-rows in the input DataFrame
    """
    def __init__(self, data, input_column_names=None, target_column_names=None,
                 diff_column_names=[],
                 train_share=(.8, 1), input_length=1, output_length=1,
                 verbose=1, batch_size=128, limit=np.inf, **kwargs):
        DataFrame = pd.read_csv(data)
        cols = list(DataFrame.columns)
        if input_column_names is None:
            input_column_names = cols
            if verbose > 0:
                print('All available columns will be used as regressors: ' + repr(cols))
        if target_column_names is None:
            target_column_names = cols
            if verbose > 0:
                print('All available columns will be predicted: ' + repr(cols))
        self.target_column_names = target_column_names
        if len(diff_column_names) > 0:
            diffs = True
            exclude_diff = [c for c in cols if c not in diff_column_names]
        else:
            diffs = False
            exclude_diff = []
        
        excluded = [c for c in cols if c not in input_column_names]
        
        super(UserGenerator, self).__init__(
            X=DataFrame, 
            train_share=train_share,
            input_length=input_length,
            output_length=output_length,
            verbose=verbose,
            limit=limit,
            batch_size=batch_size,
            excluded=excluded,
            diffs=diffs,
            exclude_diff=exclude_diff
        )
        
    def get_target_col_ids(self, cols, ids=True):
        if cols in ['default', 'all']:
            if ids:
                return np.arange(len(self.target_column_names))
            else:
                return self.target_column_names
        elif hasattr(cols, '__iter__'):
            if type(cols[0]) == str:
                return [(i if ids else c) for i, c in enumerate(self.target_column_names) if c in cols]
            elif type(cols[0]) in [int, float]:
                return [(int(i) if ids else self.target_column_names[int(i)]) for i in cols]
        raise Exception("'cols' must be iterable contatining column names or numbers. Got" + repr(cols) + ".")
            
            
def parse(argv):
    dataset = []
    data_files = os.listdir(os.path.join(WDIR, 'data'))
    print('parse, datafiles: ' + repr(data_files))
    save_file = ''
    if len(argv) > 1:
        argvv = [argv[1]]
        for arg in argv[2:]:
            if '--' in arg:
                argvv.append(arg)
            else:
                argvv[-1] += ',' + arg
        print(argvv)
        for argg in argvv:
            print(argg)
            k, v = argg.split('=')
            assert k in ['--dataset', '--save_file'], "wrong keyword: " + k
            if k == '--dataset':
                if v in ['artificial', 'lobster', 'book']:
                    dataset = [file for file in data_files if (v in file)]
                elif v == 'household':
                    dataset = [file for file in data_files if (v in file) and ('.pkl' in file)]
                    if len(dataset) == 0:
                        dataset = ['household']
                elif v == 'household_async':
                    dataset = [file for file in data_files if (v in file) and ('.pkl' in file)]
                    if len(dataset) == 0:
                        dataset = ['household_async.pkl']                
                else:
                    for file in v.split(','):
                        assert file in data_files, repr(file) + ": no such file in data directory"
                        dataset.append(file)
            else:
                assert ',' not in v, "arguments not understood: " + k + '=' + v.replace(',', ' ')
                save_file = v
    if len(dataset) == 0:
        print("no dataset specified, trying default: artificial")
        dataset = [f for f in data_files if ('artificial' in f)]    
        assert len(dataset) > 0, 'no files for aritificial dataset available in the data directory' 
    if len(save_file) == 0:
        print("no save_file specified")
        if 'async' in dataset[0]:
            save_file = os.path.join('results', 'household_async_' + argv[0][:-3].split(SEP)[-1] + '.pkl')
        elif 'household' in dataset[0]:
            save_file = os.path.join('results', 'household_' + argv[0][:-3].split(SEP)[-1] + '.pkl') #'results/cnn2.pkl' #
        elif 'artificial' in dataset[0]:
            save_file = os.path.join('results', 'artificial_' + argv[0][:-3].split(SEP)[-1] + '.pkl')
        elif 'lobster' in dataset[0]:
            save_file = os.path.join('results', 'lobster_' + argv[0][:-3].split(SEP)[-1] + '.pkl')     
        elif 'book' in dataset[0]:
            save_file = os.path.join('results', 'book_' + argv[0][:-3].split(SEP)[-1] + '.pkl')             
        else:
            raise ValueError("Wrong dataset: " + repr(dataset))
    dataset = [os.path.join('data', f) for f in dataset]
    print("datasets found: " + repr(dataset))
    print("results will be saved in " + repr(save_file))
    return dataset, save_file

    
def get_generator(dataset):
    if 'async' in dataset:
        from nnts.household import HouseholdAsynchronousGenerator as generator
    elif 'household' in dataset:
        from nnts.household import HouseholdGenerator as generator
    elif 'artificial' in dataset:
        from nnts.artificial import ArtificialGenerator as generator  
    elif 'lobster' in dataset:
        from nnts.lobster import LOBSTERGenerator as generator
    elif 'book' in dataset:
        from nnts.book import BookGenerator as generator
    else:
        raise ValueError("No data sample generator found for '%s' dataset" % dataset)
    print('using ' + repr(generator) + ' to draw samples')
    return generator

