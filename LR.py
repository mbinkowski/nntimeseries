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
    batch_size = [128, 16],
    target_cols=[['Global_active_power']],
    objective=['regr']
)
datasets = ['household.pkl']
save_file = 'results/lr.pkl' 


def LR(datasource, params):
    globals().update(params)
    G = hdu.Generator(filename=datasource, 
                      input_length=input_length, 
                      output_length=output_length, 
                      verbose=verbose)
    
    dim = G.asarray().shape[1]
    cols = [i for i, c in enumerate(G.cnames) if c in target_cols[0]]
    regr_func = utils.make_flat_regression(input_length=input_length, cols=cols)
    
    # theano.config.compute_test_value = 'off'
    # valu.tag.test_value
    inp = Input(shape=(input_length * dim,), dtype='float32', name='value_input')

    out = Dense(output_length * len(cols), activation='softmax')(inp)
    
    nn = Model(input=inp, output=out)
    
    nn.compile(optimizer=keras.optimizers.Adam(lr=.01),
               loss='mse',
               metrics=[]) 

    train_gen = G.gen('train', batch_size=batch_size, func=regr_func)
    valid_gen = G.gen('valid', batch_size=batch_size, func=regr_func)
    
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=3, 
                        verbose=1, monitor='val_loss', restore_best=False)
    
    length = input_length + output_length
    hist = nn.fit_generator(
        train_gen,
        samples_per_epoch=G.n_train - length - 1,
        nb_epoch=1000,
        callbacks=[reducer],
    #            callbacks=[callback, keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
        validation_data=valid_gen,
        nb_val_samples=G.n_all - G.n_train - length - 1,
        verbose=verbose
    )    
    return hist, nn, reducer
    
runner = utils.ModelRunner(param_dict, datasets, LR, save_file)
runner.run(log=log)