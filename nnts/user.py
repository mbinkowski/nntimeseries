"""
Utilities file.

The file contains i.a. the ModelRunner and Generator classes.
"""
from ._imports_ import *
from .config import WDIR, SEP
from . import keras_utils, utils


class UserModel(utils.Model):
    """
    Abstract class that defines the general model structure to be passed to 
    <utils.ModelRunner>.
    Classes that inherit from <nnts.utils.Model class> should implement 
    <build> method. 
    """
    def __init__(self, data, params, tensorboard_dir="." + SEP, 
                 tb_val_limit=1024):
        """
        Aruments:
            data            - correct argument to the generator object construtor
            params          - the dictionary with all of the model hyperparameters
            tensorboard_dir - directory to store TensorBoard logs
            tb_val_limit    - max number of validation samples to use by TensorBoard
        """        
        self.name = "UserModel"
        self._set_params(params)
        self.tensorboard_dir = tensorboard_dir
        self.tb_val_limit = tb_val_limit
        self.G = UserGenerator(data, 
                               input_column_names=self.input_column_names,
                               target_column_names=self.target_column_names,
                               diff_column_names=self.diff_column_names,
                               train_share=self.train_share,
                               input_length=self.input_length,
                               output_length=self.output_length,
                               batch_size=self.batch_size,
                               verbose=self.verbose,
                               limit=self.limit)
        self.idim, self.odim = self.G.get_dims(cols=self.target_cols)   
        self.nn, self.io_func, self.callbacks = self.build()
        
    def _set_params(self, params):
        self.input_column_names = None    # default input columns (None = all columns)
        self.target_column_names  = None  # default target columns (None = all columns)
        self.diff_column_names = []       # default columns to take first difference
        self.limit = np.inf
        super(UserModel, self)._set_params(params)
    
        
class UserGenerator(utils.Generator):
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
                 batch_size=128, verbose=1, limit=np.inf):
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
   