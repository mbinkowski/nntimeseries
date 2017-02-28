"""
Created on Tue Nov 29 17:59:55 2016
@author: mbinkowski

Implementation of Convolutional Neural Network grid search.
To change the model architecture edit the SOmodel function below. 
"""
from __init__ import *

log=False

param_dict = dict(
    #i/o parameters                  
    verbose = [1 + int(log)],       # verbosity
    train_share = [(.8, 1.)],       # delimeters of the training and validation shares
    input_length = [60],            # input length (1 - stateful lstm)
    output_length = [1],            # no. of timesteps to predict (only 1 impelemented)
    batch_size = [128],             # batch size
    objective=['regr'],             # only 'regr' (regression) implemented
    diffs = [False],                # if True, work on 1st difference of series instead of original
    target_cols=['default'],        # 'default' or list of names of columns to predict    
    #training_parameters
    patience = [5],                 # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [2],                # no. of learning rate reductions
    lr = [.001],                    # initial learning rate
    clipnorm = [1.0],               # max gradient norm
    #model parameters
    norm = [10],                    # max norm for fully connected top layer's weights
    filters = [16, 32],                 # no. of convolutional filters per layer
    act = ['leakyrelu'],            # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu')
    kernelsize = [3, 1, [1, 3]],    # kernel size (if list of ints passed kernel size changes successively in consecutive layers)
    poolsize = [2],                 # max pooling size
    layers_no = [10],               # no of convolutional and pooling layers
    maxpooling = [3],               # maxpool frequency
    resnet = [False],               # if True, adding vertical connections        
)
# list of datasets to test the models on
dataset = [#'data/artificialET1SS1n100000S16.csv', 'data/artificialET1SS0n100000S16.csv', 
           #'data/artificialET1SS1n100000S64.csv', 'data/artificialET1SS0n100000S64.csv',
           'data/artificialET1SS1n100000S16.csv', 'data/artificialET1SS0n10000S16.csv', 
           'data/artificialET1SS1n10000S64.csv', 'data/artificialET1SS0n10000S64.csv']
#dataset = ['household.pkl']

if 'household' in dataset[0]:
    from household_data_utils import HouseholdGenerator as generator
    save_file = 'results/household_cvi2.pkl' #'results/cnn2.pkl' #
elif 'artificial' in dataset[0]:
    from artificial_data_utils import ArtificialGenerator as generator
    save_file = 'results/' + dataset[0].split('.')[0].split('/')[1] + '_cvi2.30' #'results/cnn2.pkl' #

def CNNmodel(datasource, params):
    """
    Function defines the convolutional network structure to be passed to utils.ModelRunner
    Aruments:
        datasource  - correct argument to the generator object construtor
        params      - the dictionary with all of the model hyperparameters
    Returns:
        keras History object, keras Model object, keras_utils.LrReducer object
    """   
    globals().update(params)
    G = generator(filename=datasource, train_share=train_share,
                  input_length=input_length, 
                  output_length=output_length, verbose=verbose,
                  batch_size=batch_size, diffs=diffs)
    
    idim, odim = G.get_dims(cols=target_cols)

    # network structure definition
    inp = Input(shape=(input_length, idim), dtype='float32', name='value_input')
    outs, loop_layers = [inp], {}
    
    for j in range(layers_no):
        if (maxpooling > 0) and ((j + 1) % maxpooling == 0):
            loop_layers['maxpool' + str(j+1)] = MaxPooling1D(pool_length=poolsize,
                                                             border_mode='valid')
            outs.append(loop_layers['maxpool' + str(j+1)](outs[-1]))
        else:    
            name = 'conv' + str(j+1)
            ks = kernelsize[j % len(kernelsize)] if (type(kernelsize) == list) else kernelsize
            loop_layers[name] = Convolution1D(filters if (j < layers_no - 1) else odim, 
                                              filter_length=ks, border_mode='same', 
                                              activation='linear', name=name,
                                              W_constraint=maxnorm(norm))
            outs.append(loop_layers[name](outs[-1]))
            
            loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
            outs.append(loop_layers[name + 'BN'](outs[-1]))
            
            # residual connections
            if resnet and (maxpooling > 0) and (j > 0) and (j % maxpooling == 0):
                outs.append(merge([outs[-1], outs[-3 * (maxpooling - 1)]], mode='sum', 
                                  concat_axis=-1, name='residual' + str(j+1)))
                
            loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (act == 'leakyrelu') else Activation(act, name=name + 'act')
            outs.append(loop_layers[name + 'act'](outs[-1]))
            
            
    flat = Flatten()(outs[-1])
    out = Dense(odim * output_length, activation='linear', W_constraint=maxnorm(norm))(flat)  
    
    nn = Model(input=inp, output=out)
    
    # network training settings
    nn.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=clipnorm),
               loss='mse') 
    
    regr_func = G.make_io_func(io_form='regression', cols=target_cols)
    train_gen = G.gen('train', func=regr_func)
    valid_gen = G.gen('valid', func=regr_func)
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=reduce_nb, 
                        verbose=verbose, monitor='val_loss', restore_best=True)
    
    print('Total model parameters: %d' % utils.get_param_no(nn))
    
    hist = nn.fit_generator(
        train_gen,
        samples_per_epoch = G.n_train - G.l,
        nb_epoch=1000,
        callbacks=[reducer],
        validation_data=valid_gen,
        nb_val_samples=G.n_all - G.n_train - G.l,
        verbose=verbose
    )    
    return hist, nn, reducer

# Runs a grid search for the above model    
runner = utils.ModelRunner(param_dict, dataset, CNNmodel, save_file)
runner.run(log=log, limit=1)