# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:23:02 2017

@author: mbinkowski
"""
if __name__ == '__main__':
    from _imports_ import *
    
# Defining example data frame
# column A enumerates entries, B contains random binomial variables, 
# columns C and D contain random noise, while column E is a sum of last 10 
# values of B multiplied by D.  
df = pd.DataFrame({'A': np.arange(1000), 
                   'B': (np.random.rand(1000)> .5) * 1.0, 
                   'C': np.random.rand(1000), 
                   'D': np.random.rand(1000)})
df['E'] = df['B'] * df['D'] 
df['E'] = np.cumsum(df['E'])
df.loc[10:, 'E'] -= np.array(df['E'][:-10])
print(df.head(20))

dataset_file = os.path.join(utils.WDIR, 'data', 'example1.csv')
save_file = os.path.join(utils.WDIR, 'results', 'example_model.pkl')
df.to_csv(dataset_file)

# Defining parameters for training.
# We want to train models that predict column A given B, C and D, and A and E, 
# given B, C and D.  
log = False
param_dict = dict(
    # input parameters
    input_column_names = [['B', 'C', 'D']],    # input columns 
    target_column_names = [['E'], ['A', 'E']], # target columns
    diff_column_names = [[]],                  # columns to take first difference of   
    verbose = [1 + int(log)],   # verbosity
    train_share = [(.7, .8, 1.)],   # delimeters of the training and validation shares
    input_length = [1],         # input length (1 - stateful lstm)
    output_length = [1],        # no. of timesteps to predict (only 1 impelemented)
    batch_size = [128],         # batch size
    objective=['regr'],         # only 'regr' (regression) implemented
    diffs = [False],            # if yes, work on 1st difference of series instead of original
    target_cols=['default'],    # 'default' or list of names of columns to predict  
    #training_parameters
    patience = [10],             # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [2],            # no. of learning rate reductions
    lr = [.00001],                # initial learning rate
    clipnorm = [0.001],           # max gradient norm
    #model parameters
    norm = [1],                 # max norm for fully connected top layer's weights    
    layer_size = [64],  # size of lstm layer
    act = ['leakyrelu'],        # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu') 
    dropout = [0],              # dropout rate
    layers_no = [1],            # no. of LSTM layers
)


# defining the LSTM model class
# model class has to inherit from user.UserModel and implement 'build' method 
class LRmodel(user.UserModel):
    """
    Class defines linear regression structure to be passed to utils.ModelRunner
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
    
# Run a grid search for the above model.    
if __name__ == '__main__':
    runner = utils.ModelRunner(param_dict, [dataset_file], save_file)
    runner.run(LRmodel, log=log, limit=1)
