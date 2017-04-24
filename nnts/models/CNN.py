"""
Created on Tue Nov 29 17:59:55 2016
@author: mbinkowski

Implementation of Convolutional Neural Network grid search.
To change the model architecture edit the SOmodel function below. 
"""

log=False

param_dict = dict(
    #i/o parameters                  
    verbose = [1 + int(log)],       # verbosity
    train_share = [(.8, 1.)],       # delimeters of the training and validation shares
    input_length = [120],            # input length (1 - stateful lstm)
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
    filters = [16, 32],                 # no. of convolutional filters per layer
    act = ['leakyrelu'],            # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu')
    kernelsize = [3, 1, [1, 3]],    # kernel size (if list of ints passed kernel size changes successively in consecutive layers)
    poolsize = [2],                 # max pooling size
    layers_no = [10],               # no of convolutional and pooling layers
    maxpooling = [3],               # maxpool frequency
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
        self.name = 'CNN'
        # network structure definition
        inp = Input(shape=(self.input_length, self.idim), 
                    dtype='float32', name='value_input')
        outs, loop_layers = [inp], {}
        
        for j in range(self.layers_no):
            if (self.maxpooling > 0) and ((j + 1) % self.maxpooling == 0):
                loop_layers['maxpool' + str(j+1)] = MaxPooling1D(pool_size=self.poolsize,
                                                                 padding='valid')
                outs.append(loop_layers['maxpool' + str(j+1)](outs[-1]))
            else:    
                name = 'conv' + str(j+1)
                ks = self.kernelsize[j % len(self.kernelsize)] if (type(self.kernelsize) == list) else self.kernelsize
                loop_layers[name] = Convolution1D(self.filters if (j < self.layers_no - 1) else self.odim, 
                                                  kernel_size=ks, padding='same', 
                                                  activation='linear', name=name,
                                                  kernel_constraint=maxnorm(self.norm))
                outs.append(loop_layers[name](outs[-1]))
                
                loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
                outs.append(loop_layers[name + 'BN'](outs[-1]))
                
                # residual connections
                if self.resnet and (self.maxpooling > 0) and (j > 0) and (j % self.maxpooling == 0):
                    outs.append(keras.layers.add([outs[-1], outs[-3 * (self.maxpooling - 1)]], 
                                                      name='residual' + str(j+1)))
    #                outs.append(merge([outs[-1], outs[-3 * (maxpooling - 1)]], mode='sum', 
    #                                  concat_axis=-1, name='residual' + str(j+1)))                
                loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (self.act == 'leakyrelu') else Activation(self.act, name=name + 'act')
                outs.append(loop_layers[name + 'act'](outs[-1]))
                
                
        flat = Flatten()(outs[-1])
        out = Dense(self.odim * self.output_length, activation='linear', 
                    kernel_constraint=maxnorm(self.norm))(flat)  
        
        nn = Model(inputs=inp, outputs=out)
        
        # network training settings
        nn.compile(optimizer=keras.optimizers.Adam(lr=self.lr, 
                                                   clipnorm=self.clipnorm),
                   loss='mse') 
        
        io_func = self.G.make_io_func(io_form='regression', cols=self.target_cols)
        
        callbacks = [keras_utils.LrReducer(patience=self.patience, reduce_rate=.1, 
                                    reduce_nb=self.reduce_nb, verbose=self.verbose, 
                                    monitor='val_loss', restore_best=True)]
        return nn, io_func, callbacks

# Runs a grid search for the above model    
if __name__ == '__main__':
    dataset, save_file = utils.parse(sys.argv)
    runner = utils.ModelRunner(param_dict, dataset, save_file)
    runner.run(CNNmodel, log=log, limit=1)