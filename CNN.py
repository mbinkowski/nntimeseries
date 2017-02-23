from __init__ import *
import utils


log=True

param_dict = dict(
    verbose = [1 + int(log)],
    train_share = [(.8, 1.)],
    input_length = [60],
    output_length = [1],
    patience = [10],
    filters = [32, 16],
    act = ['linear'],
    dropout = [(0, )],#, (0, 0), (.5, 0)],
    kernelsize = [3],
    poolsize = [2],
    layers_no = [10],
    batch_size = [128],
    objective=['regr'],
    norm = [1],
    maxpooling = [3], #maxpool frequency
    resnet = [False],
    diffs = [False],               
    target_cols=['default']
)
#dataset = ['household.pkl']
dataset = ['data/artificialET1SS1n100000S16.csv', 'data/artificialET1SS0n100000S16.csv', 
           'data/artificialET1SS1n100000S64.csv', 'data/artificialET1SS0n100000S64.csv',
#           'data/artificialET1SS1n50000S64.csv', 'data/artificialET1SS0n50000S64.csv',
           'data/artificialET1SS1n10000S16.csv', 'data/artificialET1SS0n10000S16.csv',
           'data/artificialET1SS1n10000S64.csv', 'data/artificialET1SS0n10000S64.csv'
           ]#['household.pkl'] #
               
if 'household' in dataset[0]:
    from household_data_utils import HouseholdGenerator as gen
    save_file = 'results/household_cnn.pkl' #'results/cnn2.pkl' #
elif 'artificial' in dataset[0]:
    from artificial_data_utils import ArtificialGenerator as gen
    save_file = 'results/' + dataset[0].split('.')[0].split('/')[1] + '_cnn.31.pkl' #'results/cnn2.pkl' #

def CNN(datasource, params):
    globals().update(params)
    G = gen(filename=datasource, train_share=train_share,
            input_length=input_length, 
            output_length=output_length, verbose=verbose,
            batch_size=batch_size, diffs=diffs)
    
    dim = G.get_dim()
    cols = G.get_target_cols()
    regr_func = G.make_io_func(io_form='regression', cols=target_cols)

    # theano.config.compute_test_value = 'off'
    # valu.tag.test_value
    inp = Input(shape=(input_length, dim), dtype='float32', name='value_input')

    outs = [inp]
    loop_layers = {}
    
    for j in range(layers_no):
        if (maxpooling > 0) and ((j + 1) % maxpooling == 0):
            loop_layers['maxpool' + str(j+1)] = MaxPooling1D(pool_length=poolsize,
                                                             border_mode='valid')
            outs.append(loop_layers['maxpool' + str(j+1)](outs[-1]))
        else:    
            name = 'conv' + str(j+1)
            ks = kernelsize[j % len(kernelsize)] if (type(kernelsize) == list) else kernelsize
            loop_layers[name] = Convolution1D(filters if (j < layers_no - 1) else len(cols), 
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
            
            
#    mp5 = Dropout(dropout)(mp5)
    flat = Flatten()(outs[-1])
    out = Dense(len(cols) * output_length, activation='linear', W_constraint=maxnorm(100))(flat)  
    
    nn = Model(input=inp, output=out)
    
    nn.compile(optimizer=keras.optimizers.Adam(lr=.001),
               loss='mse') 

    train_gen = G.gen('train', func=regr_func)
    valid_gen = G.gen('valid', func=regr_func)
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=3, verbose=1, monitor='val_loss', restore_best=True)
    
    print('Total model parameters: %d' % int(np.sum([np.sum([np.prod(K.eval(w).shape) for w in l.trainable_weights]) for l in nn.layers])))
    
    length = input_length + output_length
    hist = nn.fit_generator(
        train_gen,
        samples_per_epoch = G.n_train - length,
        nb_epoch=1000,
        callbacks=[reducer],
    #            callbacks=[callback, keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
        validation_data=valid_gen,
        nb_val_samples=G.n_all - G.n_train - length,
        verbose=verbose
    )    
    return hist, nn, reducer
    
runner = utils.ModelRunner(param_dict, dataset, CNN, save_file)
runner.run(log=log, limit=1)
