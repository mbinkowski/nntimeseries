"""
Implementation of grid search for multi-layer LSTM model. 
To change the model architecture edit the LSTMmodel function below. 
"""

log=False                     # if True, all the verbose output is saved in logs/ 

# dictionary of hyperparameter lists across which the grid search will run
param_dict = dict(
    #input parameters
    verbose = [1 + int(log)],   # verbosity
    train_share = [(.8, .9, 1.)],   # delimeters of the training and validation shares
    input_length = [60],         # input length (1 - stateful lstm)
    output_length = [1],        # no. of timesteps to predict (only 1 impelemented)
    batch_size = [64],         # batch size
    objective=['regr'],         # only 'regr' (regression) implemented
    diffs = [False],            # if yes, work on 1st difference of series instead of original
    target_cols=[1],    # 'default' or list of names of columns to predict  
    #training_parameters
    patience = [5],             # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [3],            # no. of learning rate reductions
    lr = [.001],                # initial learning rate
    clipnorm = [0.001],           # max gradient norm
    #model parameters
    norm = [1],                 # max norm for fully connected top layer's weights    
    layer_size = [16],  # size of lstm layer
    act = ['leakyrelu'],        # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu') 
    dropout = [0.2],              # dropout rate
    layers_no = [2],            # no. of LSTM layers
)

if __name__ == '__main__':
    from _imports_ import *
else:
    from .. import *
    from ..utils import *
    
class LSTMmodel(utils.Model):
    """
    Class defines the LSTM network structure to be passed to utils.ModelRunner
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
        self.name = 'LSTM2'
        self.shuffle = True
        # network structure definition
        nn = Sequential()   
#        batch_input_shape = (self.batch_size, int(self.input_length), self.idim)
        input_shape = (int(self.input_length), self.idim)
        if self.dropout > 0:
            nn.add(Dropout(self.dropout, 
                           input_shape=input_shape,
                           name='dropout0'))
        for n in np.arange(self.layers_no - 1):
            nn.add(LSTM(self.layer_size,
                        input_shape=input_shape,
                        stateful=False, activation=None,
                        recurrent_activation='sigmoid', name='lstm%d' % (n + 1),
                        recurrent_constraint=maxnorm(self.norm),
                        kernel_constraint=maxnorm(self.norm),
                        return_sequences=True))
            if self.act == 'leakyrelu':
                nn.add(LeakyReLU(alpha=.1, name='lstm_act%d' % (n + 1)))
            else:
                nn.add(Activation(self.act, name='lstm_act%d' % (n + 1)))
            if self.dropout > 0:
                nn.add(Dropout(self.dropout, 
                               input_shape=input_shape,
                               name='dropout%d' % (n + 1)))
        nn.add(LSTM(self.odim, activation=None,
                    recurrent_activation='sigmoid', name='lstm%d' % self.layers_no,
                    recurrent_constraint=maxnorm(self.norm),
                    kernel_constraint=maxnorm(self.norm)))
        
        nn.compile(optimizer=keras.optimizers.Adam(lr=self.lr, 
                                                   clipnorm=self.clipnorm),
                   loss='mse') 
    
        # network training settings
        io_func = self.G.make_io_func(io_form='regression', 
                                      cols=self.target_cols)
        
        callbacks = [keras_utils.LrReducer(patience=self.patience, reduce_rate=.1, 
                                          reduce_nb=self.reduce_nb, 
                                          verbose=self.verbose, 
                                          monitor='val_loss', restore_best=True, 
                                          reset_states=True)]
        return nn, io_func, callbacks
        

# Runs a grid search for the above model   
if __name__ == '__main__':
    dataset, save_file = utils.parse(['LSTM2.py', '--dataset=lobster'])#sys.argv)#
    runner = utils.ModelRunner(param_dict, dataset, save_file)
    runner.run(LSTMmodel, log=log, limit=1)
