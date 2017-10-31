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
    kernelsize = [1, 3],    # kernel size (if list of ints passed kernel size changes successively in consecutive layers)
    layers_no = [4],                 # no. of layers for significance and offset sub-networks             
    architecture = [{'softmax': True, 'lambda': False}], # final activation: lambda=True indicates softplus   
    nonnegative = [False],          # if True, apply only nonnegative weights at the top layer
    connection_freq = [2],          # vertical connection frequency for ResNet
    aux_weight = [0.],    # auxilllary loss weight
    shared_final_weights = [False], # if True, same weights of timesteps for all dimentions are trained
    resnet = [False],               # if True, adds vertical connections
)

if __name__ == '__main__':
    from _imports_ import *
else:
    from .. import *
    from ..utils import *

class EDmodel(utils.Model):
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
        
        encoder, decoder, loop_layers = [inp], [], {}
        
        for j in range(self.layers_no):
            # significance sub-network
            name = 'encoder' + str(j+1)
            ks = self.kernelsize
            loop_layers[name] = Conv1D(
                self.filters, 
                kernel_size=ks, padding='same', 
                activation='linear', name=name,
                kernel_constraint=maxnorm(self.norm)
            )
            encoder.append(loop_layers[name](encoder[-1]))
            
            loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
            encoder.append(loop_layers[name + 'BN'](encoder[-1]))
            
#            # residual connections for ResNet
#            if self.resnet and (self.connection_freq > 0) and (j > 0) and ((j+1) % self.connection_freq == 0) and (j < self.layers_no['sigs'] - 1):
#                sigs.append(keras.layers.add([sigs[-1], sigs[-3 * self.connection_freq + (j==1)]], 
#                                                    name='significance_residual' + str(j+1)))
                           
            loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (self.act == 'leakyrelu') else Activation(self.act, name=name + 'act')
            encoder.append(loop_layers[name + 'act'](encoder[-1]))
        
        flat_encoded = keras.layers.Flatten()(encoder[-1])
        encoded_prediction = Dense(self.idim)(flat_encoded)
        encoded_prediction = Reshape((self.idim, 1))(encoded_prediction)
#        # permute dimenstions 
#        permuted = Permute((2, 1), 'permute_encoder')(encoder[-1])
#        if self.shared_final_weights:
#            linear = TimeDistributed(Dense(self.output_length, activation='linear', use_bias=False,
#                                           kernel_constraint=nonneg() if self.nonnegative else None),
#                                     name= 'out')
#        else: 
#            linear = LocallyConnected1D(filters=1, kernel_size=1,   # dimensions permuted. time dimension treated as separate channels, no connections between different features
#                                        padding='valid')
#        encoded_prediction = Permute((2,1), name='permute_back')(linear(permuted))
        
        decoder.append(keras.layers.concatenate([encoder[-1], encoded_prediction], axis=1))
            
        for j in range(self.layers_no):
            # offset sub-network
            name = 'decoder' + str(j+1)
            loop_layers[name] = Conv1D(
                self.filters if (j < self.layers_no['offs'] - 1) else self.idim,
                kernel_size=self.kernelsize, padding='same', 
                activation='linear', name=name,
                kernel_constraint=maxnorm(self.norm)
            )
            decoder.append(loop_layers[name](decoder[-1]))
            
            loop_layers[name + 'BN'] = BatchNormalization(name=name + 'BN')
            decoder.append(loop_layers[name + 'BN'](decoder[-1]))
            
#            # residual connections for ResNet
#            if self.resnet and (self.connection_freq > 0) and (j > 0) and ((j+1) % self.connection_freq == 0) and (j < self.layers_no['offs'] - 1):
#                decoder.append(keras.layers.add([decoder[-1], decoder[-3 * self.connection_freq + (j==1)]], 
#                                                       name='offset_residual' + str(j+1)))
                            
            loop_layers[name + 'act'] = LeakyReLU(alpha=.1, name=name + 'act') if (self.act == 'leakyrelu') else Activation(self.act, name=name + 'act')
            decoder.append(loop_layers[name + 'act'](decoder[-1]))
        
        output = decoder[-1]
            # offset -> significance connection
    #        if connection_freq > 0:
    #            if ((j+1) % connection_freq == 0) and (j+1 < layers_no):    
    #                sigs.append(merge([decoder[-1], sigs[-1]], mode='concat', concat_axis=-1, name='concat' + str(j+1)))
                
        # merging offset with appropriate dimensions of the input
        
        nn = keras.models.Model(inputs=[inp], outputs=[output])

#        for l in nn.layers:
#            print('Layer ' + l.name + ' shapes: ' + repr((l.input_shape, l.output_shape)))
        
        # network training settings
        nn.compile(optimizer=keras.optimizers.Adam(lr=self.lr, clipnorm=self.clipnorm),
                   loss={'main_output': 'mse', 'value_output' : 'mse'},
                   loss_weights={'main_output': 1., 'value_output': self.aux_weight}) 
    
        io_func = self.G.make_io_func(io_form='regression', cols=self.target_cols)    
        callbacks = [keras_utils.LrReducer(patience=self.patience, reduce_rate=.1, 
                                        reduce_nb=self.reduce_nb, verbose=self.verbose, 
                                        monitor='val_main_output_loss', 
                                        restore_best=True)]
    
        return nn, io_func, callbacks
    
# Runs a grid search for the above model    
if __name__ == '__main__':
    dataset, save_file = utils.parse(sys.argv)# ['SOCNN.py', '--dataset=household'])#
    runner = utils.ModelRunner(param_dict, dataset, save_file)
    runner.run(SOCNNmodel, log=log, limit=1)
