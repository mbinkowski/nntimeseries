"""
Implementation of Significance-Output Convolutional Neural Network. 
To change the model architecture edit the SOmodel function below. 
"""
log=False

param_dict = dict(
    # input parameters
    verbose = [1 + int(log)],       # verbosity
    train_share = [(.8, .9, 1.)],       # delimeters of the training and validation shares
    input_length = [60],            # input length (1 - stateful lstm)
    output_length = [1],            # no. of timesteps to predict (only 1 impelemented)
    batch_size = [64],             # batch size
    objective=['regr'],             # only 'regr' (regression) implemented
    diffs = [True],                # if True, work on 1st difference of series instead of original
    target_cols=['default'],        # 'default' or list of names of columns to predict    
    #training_parameters
    patience = [5],                 # no. of epoch after which learning rate will decrease if no improvement
    reduce_nb = [2],                # no. of learning rate reductions
    lr = [.01],                    # initial learning rate
    clipnorm = [1.0],               # max gradient norm
    #model parameters
    norm = [10],                    # max norm for fully connected top layer's weights
    filters = [8],                 # no. of convolutional filters per layer
    act = ['leakyrelu'],            # activation ('linear', 'relu', 'sigmoid', 'tanh', 'leakyrelu')
    kernelsize = [[1, 3], 1, 3],    # kernel size (if list of ints passed kernel size changes successively in consecutive layers)
    layers_no = [{'sigs': 10, 'offs': 1}],
#                 {'sigs': 10, 'offs': 2},
#                 {'sigs': 10, 'offs': 5},
#                 {'sigs': 10, 'offs': 10}],  # no. of layers for significance and offset sub-networks             
    architecture = [{'softmax': True, 'lambda': False}], # final activation: lambda=True indicates softplus   
    nonnegative = [False],          # if True, apply only nonnegative weights at the top layer
    connection_freq = [2],          # vertical connection frequency for ResNet
    aux_weight = [0.],    # auxilllary loss weight
    shared_final_weights = [False], # if True, same weights of timesteps for all dimentions are trained
    resnet = [False],               # if True, adds vertical connections
)

if __name__ == '__main__':
    from _imports_ import *
#else:
#    from ._imports_ import *

class SOCNNmodel(utils.Model):
    """
    Class defines the SOCNN network structure to be passed to utils.ModelRunner
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
        self.name = 'SOCNN'
        # network structure definition
        inp = Input(shape=(self.input_length, self.idim), dtype='float32', name='inp')
        value_input = Input(shape=(self.input_length, self.odim), dtype='float32', name='value_input')
        
        offsets, sigs, loop_layers = [inp], [inp], {}
        
        for j in range(self.layers_no['sigs']):
            # significance sub-network
            name = 'significance' + str(j+1)
            ks = self.kernelsize[j % len(self.kernelsize)] if (type(self.kernelsize) == list) else self.kernelsize
            loop_layers[name] = Conv1D(
                self.filters if (j < self.layers_no['sigs'] - 1) else self.odim, 
                kernel_size=ks, padding='same', 
                activation='linear', name=name,
                kernel_constraint=maxnorm(self.norm)
            )
            sigs.append(loop_layers[name](sigs[-1]))
            
            loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
            sigs.append(loop_layers[name + 'BN'](sigs[-1]))
            
            # residual connections for ResNet
            if self.resnet and (self.connection_freq > 0) and (j > 0) and ((j+1) % self.connection_freq == 0) and (j < self.layers_no['sigs'] - 1):
                sigs.append(keras.layers.add([sigs[-1], sigs[-3 * self.connection_freq + (j==1)]], 
                                                    name='significance_residual' + str(j+1)))
                           
            loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (self.act == 'leakyrelu') else Activation(self.act, name=name + 'act')
            sigs.append(loop_layers[name + 'act'](sigs[-1]))
        
        for j in range(self.layers_no['offs']):
            # offset sub-network
            name = 'offset' + str(j+1)
            loop_layers[name] = Conv1D(
                self.filters if (j < self.layers_no['offs'] - 1) else self.odim,
                kernel_size=1, padding='same', 
                activation='linear', name=name,
                kernel_constraint=maxnorm(self.norm)
            )
            offsets.append(loop_layers[name](offsets[-1]))
            
            loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
            offsets.append(loop_layers[name + 'BN'](offsets[-1]))
            
            # residual connections for ResNet
            if self.resnet and (self.connection_freq > 0) and (j > 0) and ((j+1) % self.connection_freq == 0) and (j < self.layers_no['offs'] - 1):
                offsets.append(keras.layers.add([offsets[-1], offsets[-3 * self.connection_freq + (j==1)]], 
                                                       name='offset_residual' + str(j+1)))
                            
            loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (self.act == 'leakyrelu') else Activation(self.act, name=name + 'act')
            offsets.append(loop_layers[name + 'act'](offsets[-1]))
            
            # offset -> significance connection
    #        if connection_freq > 0:
    #            if ((j+1) % connection_freq == 0) and (j+1 < layers_no):    
    #                sigs.append(merge([offsets[-1], sigs[-1]], mode='concat', concat_axis=-1, name='concat' + str(j+1)))
                
        # merging offset with appropriate dimensions of the input
        value_output = keras.layers.add([offsets[-1], value_input], name='value_output')
    
        value = Permute((2,1))(value_output)
    
        # copmuting weights from significance net
        sig = Permute((2,1))(sigs[-1])
        if self.architecture['softmax']:    
            sig = TimeDistributed(Activation('softmax'), name='softmax')(sig)
        elif self.architecture['lambda']:    
            sig = TimeDistributed(Activation('softplus'), name='relulambda')(sig)
            sig = TimeDistributed(Lambda(lambda x: x/K.sum(x, axis=-1, keepdims=True)), name='lambda')(sig)
            
        main = keras.layers.multiply(inputs=[sig, value], name='significancemerge')
        if self.shared_final_weights:
            out = TimeDistributed(Dense(self.output_length, activation='linear', use_bias=False,
                                        kernel_constraint=nonneg() if self.nonnegative else None),
                                  name= 'out')(main)
        else: 
            outL = LocallyConnected1D(filters=1, kernel_size=1,   # dimensions permuted. time dimension treated as separate channels, no connections between different features
                                      padding='valid')
            out = outL(main)
            
        main_output = Permute((2,1), name='main_output')(out)
        
        nn = Model(inputs=[inp, value_input], outputs=[main_output, value_output])
        
        # network training settings
        nn.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=self.clipnorm),
                   loss={'main_output': 'mse', 'value_output' : 'mse'},
                   loss_weights={'main_output': 1., 'value_output': self.aux_weight}) 
    
        io_func = self.G.make_io_func(io_form='cvi_regression', cols=self.target_cols)    
        callbacks = [keras_utils.LrReducer(patience=self.patience, reduce_rate=.1, 
                                        reduce_nb=self.reduce_nb, verbose=self.verbose, 
                                        monitor='val_main_output_loss', 
                                        restore_best=True)]
    
        return nn, io_func, callbacks
    
# Runs a grid search for the above model    
if __name__ == '__main__':
    dataset, save_file = utils.parse(['SOCNN.py', '--dataset=household'])#sys.argv)#
    runner = utils.ModelRunner(param_dict, dataset, save_file)
    runner.run(SOCNNmodel, log=log, limit=1)
