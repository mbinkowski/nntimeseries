from __init__ import *
os.chdir('C://Users//mbinkowski//cdsol-r-d.cluster//cdsol-r-d.machine_learning_studies//nntimeseries')
import utils


log=False

param_dict = dict(
    verbose = [1 + int(log)],
    train_share = [(.8, 1.)],
    input_length = [1],
    output_length = [1],
    patience = [5],
    layer_size = [8],
    act = ['linear'],
    dropout = [0],
    layers_no = [1],
    batch_size = [128],
    objective=['regr'],
    norm = [1],
    diffs = [False],               
    target_cols=['default']
)
dataset = ['data/artificialET1SS1n100000S16.csv', 'data/artificialET1SS0n100000S16.csv', 
           'data/artificialET1SS1n50000S64.csv', 'data/artificialET1SS0n50000S64.csv']#['household.pkl'] #
               
if 'household' in dataset[0]:
    from household_data_utils import HouseholdGenerator as gen
    save_file = 'results/household_lstm.pkl' #'results/cnn2.pkl' #
elif 'artificial' in dataset[0]:
    from artificial_data_utils import ArtificialGenerator as gen
    save_file = 'results/' + dataset[0].split('.')[0].split('/')[1] + '_lstm.pkl' #'results/cnn2.pkl' #

def LSTMmodel(datasource, params):
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
    nn = Sequential()
    
    if dropout > 0:
        nn.add(Dropout(dropout, name='dropout'))
    nn.add(LSTM(layer_size,
                batch_input_shape=(batch_size, input_length, dim),
                stateful=True, activation=None,
                inner_activation='sigmoid', name='lstm',
                return_sequences=True))
    if act == 'leakyrelu':
        nn.add(LeakyReLU(alpha=.1, name='lstm_act'))
    else:
        nn.add(Activation(act, name='lstm_act'))
    nn.add(TimeDistributed(Dense(len(cols), W_constraint=maxnorm(norm)), name='tddense'))
    nn.add(Reshape((input_length*len(cols),)))
    
    nn.compile(optimizer=keras.optimizers.Adam(lr=.0001),
               loss='mse') 

    train_gen = G.gen('train', func=regr_func, shuffle=False)
    valid_gen = G.gen('valid', func=regr_func, shuffle=False)
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=3, verbose=1, 
                        monitor='val_loss', restore_best=True, reset_states=True)
    
    print('Total model parameters: %d' % int(np.sum([np.sum([np.prod(K.eval(w).shape) for w in l.trainable_weights]) for l in nn.layers])))
    
    hist = nn.fit_generator(
        train_gen,
        samples_per_epoch = G.n_train - G.l,
        nb_epoch=1000,
        callbacks=[reducer],
    #            callbacks=[callback, keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
        validation_data=valid_gen,
        nb_val_samples=G.n_all - G.n_train,
        verbose=verbose
    )    
    return hist, nn, reducer
    
runner = utils.ModelRunner(param_dict, dataset, LSTMmodel, save_file)
runner.run(log=log, limit=1)