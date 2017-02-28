"""
Created on Tue Nov 29 17:59:55 2016
@author: mbinkowski

Implementation of Significance-Output Convolutional Neural Network. 
To change the model architecture edit the SOmodel function below. 
"""
from __init__ import *

log=False

param_dict = dict(
    # input parameters
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
    filters = [16],                 # no. of convolutional filters per layer
    act = ['leakyrelu'],            # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu')
    kernelsize = [3, 1, [1, 3]],    # kernel size (if list of ints passed kernel size changes successively in consecutive layers)
    layers_no = [{'sigs': 10, 'offs': 2}],  # no. of layers for significance and offset sub-networks             
    architecture = [{'softmax': True, 'lambda': False},
                    {'softmax': False, 'lambda': True}], # final activation: lambda=True indicates softplus   
    nonnegative = [False],          # if True, apply only nonnegative weights at the top layer
    connection_freq = [2],          # vertical connection frequency for ResNet
    aux_weight = [0., .1, 0.01],    # auxilllary loss weight
    shared_final_weights = [False], # if True, same weights of timesteps for all dimentions are trained
    resnet = [False],               # if True, adds vertical connections
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

def SOmodel(datasource, params):
    """
    Function defines the SOCNN network structure to be passed to utils.ModelRunner
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
    inp = Input(shape=(input_length, idim), dtype='float32', name='inp')
    value_input = Input(shape=(input_length, odim), dtype='float32', name='value_input')
    
    offsets, sigs, loop_layers = [inp], [inp], {}
    
    for j in range(layers_no['sigs']):
        # significance sub-network
        name = 'significance' + str(j+1)
        ks = kernelsize[j % len(kernelsize)] if (type(kernelsize) == list) else kernelsize
        loop_layers[name] = Convolution1D(filters if (j < layers_no['sigs'] - 1) else odim, 
                                          filter_length=ks, border_mode='same', 
                                          activation='linear', name=name,
                                          W_constraint=maxnorm(norm))
        sigs.append(loop_layers[name](sigs[-1]))
        
        loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
        sigs.append(loop_layers[name + 'BN'](sigs[-1]))
        
        # residual connections for ResNet
        if resnet and (connection_freq > 0) and (j > 0) and ((j+1) % connection_freq == 0):
            sigs.append(merge([sigs[-1], sigs[-3 * connection_freq + (j==1)]], mode='sum', 
                               concat_axis=-1, name='significance_residual' + str(j+1)))
                       
        loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (act == 'leakyrelu') else Activation(act, name=name + 'act')
        sigs.append(loop_layers[name + 'act'](sigs[-1]))
    
    for j in range(layers_no['offs']):
        # offset sub-network
        name = 'offset' + str(j+1)
        loop_layers[name] = Convolution1D(filters if (j < layers_no['offs'] - 1) else odim,
                                          filter_length=1, border_mode='same', 
                                          activation='linear', name=name,
                                          W_constraint=maxnorm(norm))
        offsets.append(loop_layers[name](offsets[-1]))
        
        loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
        offsets.append(loop_layers[name + 'BN'](offsets[-1]))
        
        # residual connections for ResNet
        if resnet and (connection_freq > 0) and (j > 0) and ((j+1) % connection_freq == 0):
            offsets.append(merge([offsets[-1], offsets[-3 * connection_freq + (j==1)]], mode='sum', 
                                  concat_axis=-1, name='offset_residual' + str(j+1)))
                        
        loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (act == 'leakyrelu') else Activation(act, name=name + 'act')
        offsets.append(loop_layers[name + 'act'](offsets[-1]))
        
        # offset -> significance connection
#        if connection_freq > 0:
#            if ((j+1) % connection_freq == 0) and (j+1 < layers_no):    
#                sigs.append(merge([offsets[-1], sigs[-1]], mode='concat', concat_axis=-1, name='concat' + str(j+1)))
            
    # merging offset with appropriate dimensions of the input
    value_output = merge([offsets[-1], value_input], mode='sum', concat_axis=-1, name='value_output')

    value = Permute((2,1))(value_output)

    # copmuting weights from significance net
    sig = Permute((2,1))(sigs[-1])
    if architecture['softmax']:    
        sig = TimeDistributed(Activation('softmax'), name='softmax')(sig)
    elif architecture['lambda']:    
        sig = TimeDistributed(Activation('softplus'), name='relulambda')(sig)
        sig = TimeDistributed(Lambda(lambda x: x/K.sum(x, axis=-1, keepdims=True)), name='lambda')(sig)
        
    main = merge([sig, value], mode='mul', concat_axis=-1, name='significancemerge')
    if shared_final_weights:
        out = TimeDistributed(Dense(output_length, activation='linear', bias=False,
                                    W_constraint=nonneg() if nonnegative else None),
                              name= 'out')(main)
    else: 
        out = LocallyConnected1D(nb_filter=1, filter_length=1,   # dimensions permuted. time dimension treated as separate channels, no connections between different features
                                 border_mode='valid')(main)
        
    main_output = Permute((2,1), name='main_output')(out)
    
    nn = Model(input=[inp, value_input], output=[main_output, value_output])
    
    # network training settings
    nn.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=clipnorm),
               loss={'main_output': 'mse', 'value_output' : 'mse'},
               loss_weights={'main_output': 1., 'value_output': aux_weight}) 

    regr_func = G.make_io_func(io_form='cvi_regression', cols=target_cols)    
    train_gen = G.gen('train', func=regr_func)
    valid_gen = G.gen('valid', func=regr_func)
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=reduce_nb, 
                        verbose=verbose, monitor='val_main_output_loss', restore_best=True)
    
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
runner = utils.ModelRunner(param_dict, dataset, SOmodel, save_file)
runner.run(log=log, limit=1, read_file='/results/all_results.pkl')