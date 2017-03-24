"""
Created on Tue Nov 29 17:59:55 2016
@author: mbinkowski

Implementation of Linear Regression model. 
To change the model architecture edit the LRmodel function below. 
"""

log=False                       # if True, all the verbose output is saved in logs/ 

# dictionary of hyperparameter lists across which the grid search will run
param_dict = dict(
    #input parameters                  
    verbose = [1 + int(log)],   # verbosity
    train_share = [(.8, 1.)],   # delimeters of the training and validation shares
    input_length = [60],        # input length 
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
)

if __name__ == '__main__':
    from _imports_ import *

def LRmodel(datasource, params):
    """
    Function defines the Linear Regression model structure to be passed to 
    utils.ModelRunner
    Aruments:
        datasource  - correct argument to the generator object construtor
        params      - the dictionary with all of the model hyperparameters
    Returns:
        keras History object, keras Model object, keras_utils.LrReducer object
    """    
    globals().update(params)
    generator = utils.get_generator(datasource)
    G = generator(filename=datasource, train_share=train_share,
                  input_length=input_length, 
                  output_length=output_length, verbose=verbose,
                  batch_size=batch_size, diffs=diffs)
    
    idim, odim = G.get_dims(cols=target_cols)

    inp = Input(shape=(input_length * idim,), dtype='float32', name='value_input')
    out = Dense(output_length * odim, activation='softmax')(inp)
    nn = Model(inputs=inp, outputs=out)
 
    # training settings
    nn.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=clipnorm),
               loss='mse',
               metrics=[]) 

    regr_func = G.make_io_func(io_form='flat_regression', cols=target_cols)
    train_gen = G.gen('train', func=regr_func)
    valid_gen = G.gen('valid', func=regr_func)
    
    reducer = keras_utils.LrReducer(patience=patience, reduce_rate=.1, 
                                    reduce_nb=reduce_nb, verbose=verbose, 
                                    monitor='val_loss', restore_best=False)
    
    print('Total model parameters: %d' % utils.get_param_no(nn))
    
    hist = nn.fit_generator(
        train_gen,
        steps_per_epoch = (G.n_train - G.l) / batch_size,
        epochs=1000,
        callbacks=[reducer],
        validation_data=valid_gen,
        validation_steps=(G.n_all - G.n_train - G.l) / batch_size,
        verbose=verbose
    )    
    return hist, nn, reducer

# Runs a grid search for the above model    
if __name__ == '__main__':
    dataset, save_file = utils.parse(sys.argv)
    runner = utils.ModelRunner(param_dict, dataset, LRmodel, save_file)
    runner.run(log=log, limit=1)