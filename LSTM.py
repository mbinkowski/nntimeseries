"""
Created on Tue Nov 29 17:59:55 2016
@author: mbinkowski

Implementation of grid search for LSTM model. 
To change the model architecture edit the LSTMmodel function below. 
"""
from __init__ import *


log=False                       # if True, all the verbose output is saved in logs/ 

# dictionary of hyperparameter lists across which the grid search will run
param_dict = dict(
    #input parameters
    verbose = [1 + int(log)],   # verbosity
    train_share = [(.8, 1.)],   # delimeters of the training and validation shares
    input_length = [1],         # input length (1 - stateful lstm)
    output_length = [1],        # no. of timesteps to predict (only 1 impelemented)
    batch_size = [128],         # batch size
    objective=['regr'],         # only 'regr' (regression) implemented
    diffs = [False],            # if yes, work on 1st difference of series instead of original
    target_cols=['default'],    # 'default' or list of names of columns to predict  
    #training_parameters
    patience = [5],             # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [2],            # no. of learning rate reductions
    lr = [.001],                # initial learning rate
    clipnorm = [1.0],           # max gradient norm
    #model parameters
    norm = [1],                 # max norm for fully connected top layer's weights    
    layer_size = [16, 32, 64],  # size of lstm layer
    act = ['leakyrelu'],        # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu') 
    dropout = [0],              # dropout rate
    layers_no = [1],            # no. of LSTM layers
)

# list of datasets to test the models on
dataset = [#'data/artificialET1SS1n100000S16.csv', 'data/artificialET1SS0n100000S16.csv', 
           #'data/artificialET1SS1n100000S64.csv', 'data/artificialET1SS0n100000S64.csv',
           'data/artificialET1SS1n100000S16.csv', 'data/artificialET1SS0n10000S16.csv', 
           'data/artificialET1SS1n10000S64.csv', 'data/artificialET1SS0n10000S64.csv']
               
if 'household' in dataset[0]:
    from household_data_utils import HouseholdGenerator as generator
    save_file = 'results/household_lstm.pkl' #'results/cnn2.pkl' #
elif 'artificial' in dataset[0]:
    from artificial_data_utils import ArtificialGenerator as generator
    save_file = 'results/' + dataset[0].split('.')[0].split('/')[1] + '_lstm_clip1.pkl'

def LSTMmodel(datasource, params):
    """
    Function defines the LSTM network structure to be passed to utils.ModelRunner
    Aruments:
        datasource  - correct argument to the generator object construtor
        params      - the dictionary with all of the model hyperparameters
    Returns:
        keras History object, keras Model object, keras_utils.LrReducer object
    """
    globals().update(params) # thanks to these we can freely use the hyperparameters by their names
    G = generator(filename=datasource, train_share=train_share,
                  input_length=input_length, 
                  output_length=output_length, verbose=verbose,
                  batch_size=batch_size, diffs=diffs)
    
    idim, odim = G.get_dims(cols=target_cols)

    # network structure definition
    nn = Sequential()    
    if dropout > 0:
        nn.add(Dropout(dropout, name='dropout'))
    nn.add(LSTM(layer_size,
                batch_input_shape=(batch_size, input_length, idim),
                stateful=True, activation=None,
                inner_activation='sigmoid', name='lstm',
                return_sequences=True))
    if act == 'leakyrelu':
        nn.add(LeakyReLU(alpha=.1, name='lstm_act'))
    else:
        nn.add(Activation(act, name='lstm_act'))
    nn.add(TimeDistributed(Dense(odim, W_constraint=maxnorm(norm)), name='tddense'))
    nn.add(Reshape((input_length * odim,)))
    
    nn.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=clipnorm),
               loss='mse') 

    # network training settings
    regr_func = G.make_io_func(io_form='regression', cols=target_cols)
    train_gen = G.gen('train', func=regr_func, shuffle=False)
    valid_gen = G.gen('valid', func=regr_func, shuffle=False)
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=reduce_nb, 
                        verbose=verbose, monitor='val_loss', 
                        restore_best=True, reset_states=True)
    
    print('Total model parameters: %d' % utils.get_param_no(nn))
    
    hist = nn.fit_generator(
        train_gen,
        samples_per_epoch = G.n_train - G.l,
        nb_epoch=1000,
        callbacks=[reducer],
        validation_data=valid_gen,
        nb_val_samples=G.n_all - G.n_train,
        verbose=verbose
    )    
    return hist, nn, reducer

# Runs a grid search for the above model
runner = utils.ModelRunner(param_dict, dataset, LSTMmodel, save_file)
runner.run(log=log, limit=2)
