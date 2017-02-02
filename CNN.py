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
    filters = [16],
    act = ['linear'],
    dropout = [(0, 0)],#, (0, 0), (.5, 0)],
    kernelsize = [3],
    poolsize = [2],
    batch_size = [16, 128],
    target_cols=[['Global_active_power']],
    objective=['regr'],
    norm = [1]
)

datasets = ['household.pkl']
save_file = 'results/cnn.pkl' 

def LR(datasource, params):
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
    conv1 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, input_shape=(input_length, dim), W_constraint=maxnorm(norm))(inp)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    conv1 = Dropout(dropout[0])(conv1)
    conv1 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, W_constraint=maxnorm(norm))(conv1)#(None, 100, 32)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=.1)(conv1)
    mp1 = MaxPooling1D(pool_length=poolsize, border_mode='valid')(conv1)
    conv2 = Dropout(dropout[1])(mp1)
#    conv2 = BatchNormalization()(conv2)
#    conv2 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, input_shape=(length, dim), W_constraint=maxnorm(norm))(conv2)
#    conv2 = LeakyReLU(alpha=.1)(conv2)
#    conv2 = Dropout(dropout[0])(conv2)
#    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, W_constraint=maxnorm(norm))(conv2)#(None, 50, 32)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=.1)(conv2)
    mp2 = MaxPooling1D(pool_length=poolsize, border_mode='valid')(conv2)
    conv3 = Dropout(dropout[1])(mp2)
#    conv3 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, input_shape=(length, dim), W_constraint=maxnorm(norm))(conv3)
#    conv3 = LeakyReLU(alpha=.1)(conv3)
#    conv3 = Dropout(dropout[0])(conv3)
#    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, W_constraint=maxnorm(norm))(conv3) #(None, 25, 32)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=.1)(conv3)
    mp3 = MaxPooling1D(pool_length=poolsize, border_mode='valid')(conv3)
    conv4 = Dropout(dropout[1])(mp3)
    conv4 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, W_constraint=maxnorm(norm))(conv4) #(None, 25, 32)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=.1)(conv4)
#    conv4 = Dropout(dropout[0])(conv4)
#    mp4 = MaxPooling1D(pool_length=2, border_mode='valid')(conv4)
    conv5 = Convolution1D(filters, kernelsize, border_mode='same', activation=act, W_constraint=maxnorm(norm))(conv4) #(None, 25, 32)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=.1)(conv5)
    mp5 = MaxPooling1D(pool_length=poolsize, border_mode='valid')(conv5)
    mp5 = Dropout(dropout[1])(mp5)
#    mp5 = Dropout(dropout)(mp5)
    flat = Flatten()(mp5)
    out = Dense(1, activation='linear', W_constraint=maxnorm(100))(flat)  
    
    nn = Model(input=inp, output=out)
    
    nn.compile(optimizer=keras.optimizers.Adam(lr=.01),
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
    
runner = utils.ModelRunner(param_dict, datasets, LR, save_file)
runner.run(log=log)