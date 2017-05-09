"""
Created on Tue Nov 29 17:59:55 2016
@author: mbinkowski

Implementation of Linear Regression model. 
To change the model architecture edit the LRmodel function below. 
"""

log=False                       # if True, all the verbose output is saved in logs/ 

# dictionary of hyperparameter lists across which the grid search will run
param_dict = dict(
    #input parameters                  
    verbose = [1 + int(log)],   # verbosity
    train_share = [(.7, .8, 1.)],   # delimeters of the training and validation shares
    input_length = [60],        # input length 
    output_length = [1],        # no. of timesteps to predict (only 1 impelemented)
    batch_size = [128],         # batch size
    objective=['regr'],         # only 'regr' (regression) implemented
    diffs = [False],            # if yes, work on 1st difference of series instead of original
    target_cols=['default'],    # 'default' or list of names of columns to predict
    #training_parameters
    patience = [10],             # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [2],            # no. of learning rate reductions
    lr = [.001],                # initial learning rate
    clipnorm = [1.0],           # max gradient norm
)

if __name__ == '__main__':
    from _imports_ import *

class LRmodel(utils.Model):
    """
    Class defines the Linear Regression model structure to be passed to 
    utils.ModelRunner.
    """  
    def build(self):
        """
        Function has to return:
            nn                 - keras.models.Model object
            io_func            - function that converts raw array to the input 
                                 form that feeds the model. Can be obtained through
                                 nnts.utils.Generator.make_io_func method
            callbacks          - list of keras.callbacks.Callback objects
        """
        self.name = "LR"
        # network architecture
        inp = Input(shape=(self.input_length * self.idim,), dtype='float32', 
                    name='value_input')
        out = Dense(self.output_length * self.odim, activation='softmax')(inp)
        
        nn = Model(inputs=inp, outputs=out)
     
        # training settings
        nn.compile(optimizer=keras.optimizers.Adam(lr=self.lr, 
                                                   clipnorm=self.clipnorm),
                   loss='mse', metrics=[]) 
    
        io_func = self.G.make_io_func(io_form='flat_regression', 
                                      cols=self.target_cols)
    
        callbacks = [keras_utils.LrReducer(patience=self.patience, reduce_rate=.1, 
                                    reduce_nb=self.reduce_nb, verbose=self.verbose, 
                                    monitor='val_loss', restore_best=False)]
        return nn, io_func, callbacks

# Runs a grid search for the above model    
if __name__ == '__main__':
    dataset, save_file = utils.parse(sys.argv)
    runner = utils.ModelRunner(param_dict, dataset, save_file)
    runner.run(LRmodel, log=log, limit=1)