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
    batch_size = [128],
    target_cols=[['Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
                  'Sub_metering_3']],
    objective=['regr']
)
datasets = ['household.pkl']
save_file = 'results/lr.pkl' 
gen = hdu.HouseholdGenerator

def LR(datasource, params):
    globals().update(params)
    G = gen(filename=datasource, input_length=input_length, 
            output_length=output_length, verbose=verbose, 
            batch_size=batch_size)
    
    dim = G.asarray().shape[1]
    cols = [i for i, c in enumerate(G.cnames) if c in target_cols]
    regr_func = G.make_io_func(io_form='flat_regression', cols=cols)
    
    # theano.config.compute_test_value = 'off'
    # valu.tag.test_value
    inp = Input(shape=(input_length * dim,), dtype='float32', name='value_input')

    out = Dense(output_length * len(cols), activation='softmax')(inp)
    
    nn = Model(input=inp, output=out)
    
    nn.compile(optimizer=keras.optimizers.Adam(lr=.01),
               loss='mse',
               metrics=[]) 

    train_gen = G.gen('train', func=regr_func)
    valid_gen = G.gen('valid', func=regr_func)
    
    reducer = LrReducer(patience=patience, reduce_rate=.1, reduce_nb=2, 
                        verbose=1, monitor='val_loss', restore_best=False)
    
    length = input_length + output_length
    hist = nn.fit_generator(
        train_gen,
        samples_per_epoch=G.n_train - length,
        nb_epoch=1000,
        callbacks=[reducer],
    #            callbacks=[callback, keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)],
        validation_data=valid_gen,
        nb_val_samples=G.n_all - G.n_train - length,
        verbose=verbose
    )    
    return hist, nn, reducer
    
runner = utils.ModelRunner(param_dict, datasets, LR, save_file)
runner.run(log=log)