from __init__ import *
os.chdir('C://Users//mbinkowski//cdsol-r-d.cluster//cdsol-r-d.machine_learning_studies//nntimeseries')
import utils
import household_data_utils as hdu

log=True

param_dict = dict(
    verbose = [1 + int(log)],
    input_length = [60],
    output_length = [1],
    patience = [5],
    filters = [8],
    act = ['linear'],
    dropout = [(0, 0)],#, (0, 0), (.5, 0)],
    kernelsize = [1, [1, 3], 3],
    layers_no = [10],
    poolsize = [None],
    architecture = [{'softmax': True, 'lambda': False, 'nonneg': False}],
    batch_size = [128],
    target_cols=[['Global_active_power']],
    objective=['regr'],
    norm = [1],
    nonnegative = [False],
    train_offset = [False, True]
)

datasets = ['household.pkl']
save_file = 'results/cnnvi.pkl' 

def CNN(datasource, params):
    globals().update(params)
    G = hdu.Generator(filename=datasource, 
                      input_length=input_length, 
                      output_length=output_length, 
                      verbose=verbose)
    
    dim = G.asarray().shape[1]
    cols = [i for i, c in enumerate(G.cnames) if c in target_cols[0]]
    regr_func = utils.make_vi_regression(input_length=input_length, cols=cols)

    inp = Input(shape=(input_length, dim), dtype='float32', name='inp')
    value_input = Input(shape=(input_length, len(cols)), dtype='float32', name='value_input')
    
    offsets = [inp]
    sigs = [inp]
    loop_layers = {}
    
    for j in range(layers_no):
        name = 'significance' + str(j+1)
        ks = kernelsize[j % len(kernelsize)] if (type(kernelsize) == list) else kernelsize
        loop_layers[name] = Convolution1D(filters if (j < layers_no - 1) else len(cols), 
                                          filter_length=ks, border_mode='same', 
                                          activation='linear', name=name,
                                          W_constraint=maxnorm(norm))
        sigs.append(loop_layers[name](sigs[-1]))
        
        loop_layers[name + 'BN'] = BatchNormalization()
        sigs.append(loop_layers[name + 'BN'](sigs[-1]))
                       
        loop_layers[name + 'act'] = LeakyReLU(alpha=.1) if (act == 'leakyrelu') else Activation(act)
        sigs.append(loop_layers[name + 'act'](sigs[-1]))

    if train_offset:
        for j in range(layers_no):
            name = 'offset' + str(j+1)
            ks = kernelsize[j % len(kernelsize)] if (type(kernelsize) == list) else kernelsize
            loop_layers[name] = Convolution1D(filters if (j < layers_no - 1) else len(cols), 
                                              filter_length=ks, border_mode='same', 
                                              activation='linear', name=name,
                                              W_constraint=maxnorm(norm))
            offsets.append(loop_layers[name](offsets[-1]))
            
            loop_layers[name + 'BN'] = BatchNormalization()
            offsets.append(loop_layers[name + 'BN'](offsets[-1]))
                           
            loop_layers[name + 'act'] = LeakyReLU(alpha=.1) if (act == 'leakyrelu') else Activation(act)
            offsets.append(loop_layers[name + 'act'](offsets[-1]))
                           
        value = merge([offsets[-1], value_input], mode='sum', concat_axis=-1, name='offsetmerge')
    else: 
        value = value_input
    value = Permute((2,1))(value)

    sig = Permute((2,1))(sigs[-1])
    sig = TimeDistributed(Dense(input_length, activation='softmax'), name='softmax')(sig)
    
    main = merge([sig, value], mode='mul', concat_axis=-1, name='significancemerge')
    out = TimeDistributed(Dense(output_length, activation='linear', bias=False,
                                W_constraint=nonneg() if nonnegative else None))(main)
    out = Permute((2,1))(out)
    
    nn = Model(input=[inp, value_input], output=out)
    
    nn.compile(optimizer=keras.optimizers.Adam(lr=.001),
               loss='mse') 

    train_gen = G.gen('train', batch_size=batch_size, func=regr_func)
    valid_gen = G.gen('valid', batch_size=batch_size, func=regr_func)
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=3, verbose=1, monitor='val_loss', restore_best=True)
    
    length = input_length + output_length
    hist = nn.fit_generator(
        train_gen,
        samples_per_epoch = G.n_train - length-1,
        nb_epoch=1000,
        callbacks=[reducer],
    #            callbacks=[callback, keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
        validation_data=valid_gen,
        nb_val_samples=G.n_all - G.n_train - length-1,
        verbose=verbose
    )    
    return hist, nn, reducer
    
runner = utils.ModelRunner(param_dict, datasets, CNN, save_file)
runner.run(log=log)