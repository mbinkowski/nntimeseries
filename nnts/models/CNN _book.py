"""
Implementation of Convolutional Neural Network grid search.
To change the model architecture edit the CNNmodel function below. 
"""

log=False

param_dict = dict(
    #i/o parameters                  
    verbose = [1 + int(log)],       # verbosity
    train_share = [(.8, .9, 1.)],       # delimeters of the training and validation shares
    input_length = [1024],            # input length (1 - stateful lstm)
    output_length = [256],            # no. of timesteps to predict (only 1 impelemented)
    batch_size = [64],             # batch size
    objective=['regr'],             # only 'regr' (regression) implemented
    diffs = [False],                # if True, work on 1st difference of series instead of original
    target_cols=['default'],        # 'default' or list of names of columns to predict    
    #training_parameters
    patience = [5],                 # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [2],                # no. of learning rate reductions
    lr = [.001],                    # initial learning rate
    clipnorm = [1.0],               # max gradient norm
    #model parameters
    dropout = [.2, .5],          # dropout
    norm = [10],                    # max norm for fully connected top layer's weights
    filters = [16],                 # no. of convolutional filters per layer
    act = ['leakyrelu'],            # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu')
    kernelsize = [3],    # kernel size (if list of ints passed kernel size changes successively in consecutive layers)
    poolsize = [2],                 # max pooling size
    layers_no = [10],               # no of convolutional and pooling layers
    maxpooling = [2],               # maxpool frequency
    resnet = [False],               # if True, adding vertical connections        
)

if __name__ == '__main__':  
    from _imports_ import *

class CNNmodel(utils.Model):
    """
    Class defines the convolutional network structure to be passed to utils.ModelRunner
    """   
    def build(self):
        """
        Function has to return:
            nn                 - keras.models.Model object
            io_func            - function that converts raw array to the input 
                                 form that feeds the model. Can be obtained through
                                 nnts.utils.Generator.make_io_func method
            callbacks          - list of keras.callbacks.Callback objects
        """     
        self.name = 'CNN_book'
        # network structure definition
        inp = Input(shape=(self.input_length, self.idim), 
                    dtype='float32', name='value_input')
        outs = [inp]
        
        for j in range(self.layers_no):
            if (self.maxpooling > 0) and ((j + 1) % self.maxpooling == 0):
                outs.append(MaxPooling1D(
                    pool_size=self.poolsize,
                    padding='valid',
                    name='%d_pool' % (j+1)
                )(outs[-1]))
            else: 
                name = '%d_' % (j+1)
                if self.dropout > 0:
                    outs.append(Dropout(self.dropout, name=name + 'dropout')(outs[-1]))
                ks = self.kernelsize[j % len(self.kernelsize)] if (type(self.kernelsize) == list) else self.kernelsize
                outs.append(Conv1D(
                    self.filters, 
                    kernel_size=ks, padding='same', 
                    activation='linear', name=name + 'conv',
                    kernel_constraint=maxnorm(self.norm)
                )(outs[-1]))
                
                outs.append(BatchNormalization(name=name + 'BN')(outs[-1]))
                outs.append(keras_utils.Activation_(self.act, name)(outs[-1]))

                if self.resnet and (j > 0) and (j < self.layers_no - 1):
                    outs.append(keras.layers.add([outs[-1], outs[-2]], name=name + 'residual'))
        flat = Flatten()(outs[-1])
        
        n_exp_times, t = 1, self.output_length
        while t >= 1:
            n_exp_times += 1
            t = t//2
        
        dense = Dense(n_exp_times * 2, activation='linear', 
                    kernel_constraint=maxnorm(self.norm))(flat) 
        out = Reshape((n_exp_times, 2))(dense)
#        TODO: output sizes below

        
        nn = Model(inputs=inp, outputs=out)
        for l in nn.layers:
            print('Layer ' + l.name + ' shapes: ' + repr((l.input_shape, l.output_shape)))
        # network training settings
        loss = nnts.book.def_pnl_loss_L2(.2)
        metrics = [nnts.book.def_pnl(i) for i in range(1, 10)] + ['mse']
        nn.compile(optimizer=keras.optimizers.Adam(lr=self.lr, 
                                                   clipnorm=self.clipnorm),
                   loss=loss,
                   metrics=metrics) 
        
        io_func = self.G.make_io_func(io_form='0+exp_time', cols=self.target_cols)
        
        callbacks = [keras_utils.LrReducer(patience=self.patience, reduce_rate=.1, 
                                    reduce_nb=self.reduce_nb, verbose=self.verbose, 
                                    monitor='val_loss', restore_best=True)]
        return nn, io_func, callbacks

# Runs a grid search for the above model    
if __name__ == '__main__':
    dataset, save_file = utils.parse(['CNN_book.py', '--dataset=book'])#
    runner = utils.ModelRunner(param_dict, dataset, save_file)
    runner.run(CNNmodel, log=log, limit=1)