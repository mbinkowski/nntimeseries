from __init__ import *
os.chdir('C://Users//mbinkowski//cdsol-r-d.cluster//cdsol-r-d.machine_learning_studies//nntimeseries')
import utils
import household_data_utils as hdu

log=False

param_dict = dict(
    verbose = [1 + int(log)],
    input_length = [60],
    output_length = [1],
    patience = [5],
    filters = [16],
    act = ['linear'],
    dropout = [(0, )],#, (0, 0), (.5, 0)],
    kernelsize = [3],
    poolsize = [2],
    layers_no = [10],
    batch_size = [128],
    target_cols=[['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
                  'Sub_metering_3']],
    objective=['regr'],
    norm = [1],
    maxpooling = [0, 3, 5], #maxpool frequency
    resnet = [False]
)

datasets = ['household.pkl']
save_file = 'results/cnn.pkl' 

def CNN(datasource, params):
    globals().update(params)
    G = hdu.Generator(filename=datasource, 
                      input_length=input_length, 
                      output_length=output_length, 
                      verbose=verbose)
    
    dim = G.asarray().shape[1]
    cols = [i for i, c in enumerate(G.cnames) if c in target_cols[0]]
    regr_func = utils.make_regression(input_length=input_length, cols=cols)

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
            loop_layers[name] = Convolution1D(filters if (j < layers_no - 1) else len(cols), 
                                              kernelsize, border_mode='same', 
                                              activation='linear', name=name,
                                              W_constraint=maxnorm(norm))
            outs.append(loop_layers[name](outs[-1]))
            
            loop_layers[name + 'BN'] = BatchNormalization()
            outs.append(loop_layers[name + 'BN'](outs[-1]))
                           
            loop_layers[name + 'act'] = LeakyReLU(alpha=.1) if (act == 'leakyrelu') else Activation(act)
            outs.append(loop_layers[name + 'act'](outs[-1]))
            if resnet and (maxpooling > 0) and (j > 0) and (j % maxpooling == 0):
                outs.append(merge([outs[-1], outs[-3 * (maxpooling - 1)]], mode='sum', 
                                  concat_axis=-1, name='residual' + str(j+1)))
            
#    mp5 = Dropout(dropout)(mp5)
    flat = Flatten()(outs[-1])
    out = Dense(len(cols) * output_length, activation='linear', W_constraint=maxnorm(100))(flat)  
    
    nn = Model(input=inp, output=out)
    
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