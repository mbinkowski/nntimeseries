"""
Utilities file.

The file contains i.a. the ModelRunner and Generator classes.
"""
#if __name__ == '__main__': 
#    from ._imports_ import *
#    from .config import WDIR, SEP
#    from . import keras_utils, utils
#

#class UserModel(utils.Model):
#    """
#    Abstract class that defines the general model structure to be passed to 
#    <utils.ModelRunner>.
#    Classes that inherit from <nnts.utils.Model class> should implement 
#    <build> method. 
#    """
#    def __init__(self, data, params, tensorboard_dir="." + SEP, 
#                 tb_val_limit=1024):
#        """
#        Aruments:
#            data            - correct argument to the generator object construtor
#            params          - the dictionary with all of the model hyperparameters
#            tensorboard_dir - directory to store TensorBoard logs
#            tb_val_limit    - max number of validation samples to use by TensorBoard
#        """        
#        self.name = "UserModel"
#        self._set_params(params)
#        self.tensorboard_dir = tensorboard_dir
#        self.tb_val_limit = tb_val_limit
#        self.G = UserGenerator(data, 
#                               input_column_names=params['input_column_names'],
#                               target_column_names=params['target_column_names'],
#                               diff_column_names=params['diff_column_names'],
#                               train_share=self.train_share,
#                               input_length=self.input_length,
#                               output_length=self.output_length,
#                               verbose=self.verbose,
#                               batch_size=self.batch_size,
#                               limit=self.limit)
#        self.idim, self.odim = self.G.get_dims(cols=self.target_cols)   
#        self.nn, self.io_func, self.callbacks = self.build()
#        
#    def _set_params(self, params):
##        self.input_column_names = None    # default input columns (None = all columns)
##        self.target_column_names  = None  # default target columns (None = all columns)
##        self.diff_column_names = []       # default columns to take first difference
#        self.limit = np.inf
#        super(UserModel, self)._set_params(params)
    
        

   