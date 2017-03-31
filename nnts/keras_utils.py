# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:31:16 2016

@author: mbinkowski
"""
from ._imports_ import *

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1.0e-9
    clipped_pred = T.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * T.log(clipped_pred)).sum(axis=-1)
#    loss = losses.sum(axis=1)
    return loss
 
def get_multitask_cross_entropy(weights=np.array([[[1] * 3]])):
    def mce(y_true, y_pred):
        epsilon = 1.0e-9
        clipped_pred = T.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(y_true * T.log(clipped_pred) * weights).sum(axis=-1).mean(axis=-1)
        return loss
    return mce
    
def multitask_accuracies(tasks, mask=np.array([[1] * 3])):
    def make_acc(i, mask):
        def acc(y_true, y_pred):
#            return T.sum(y_true)
            y_t = y_true[:, i, :]
            y_p = T.eq(y_pred[:, i, :], y_pred[:, i, :].max(axis=-1, keepdims=True))
            return T.sum(y_t * y_p * mask) #/ K.sum(mask * y_t).clip(1, np.inf)
#            return K.mean(T.eq(K.argmax(y_true[:, i, :], axis=-1),
#                               K.argmax(y_pred[:, i, :], axis=-1)))
        return acc
    metrics = []
    for i in np.arange(tasks):
        globals()['accuracy_%d' % i] = make_acc(i, mask)
        metrics.append(globals()['accuracy_%d' % i])
    return metrics


class MyCallback(keras.callbacks.Callback):
    def __init__(self, model, data, _type='classifier', message='', 
                 batch_size=16, dump_file='nn_results.pickle',
                 monitor_activations=True, interactive_display=True,
                 mask=None):
        self.last_logs = None
        self.nn = model
        self.data = data
        self.message = message
        self.batch_size = batch_size
        self.dump_file = dump_file
        self.improvement = [1] * 5
        self.secondary = 'R2' if (_type == 'regressor') else 'accuracy'
        self.monitor_activations = monitor_activations
        if monitor_activations:
            discarded = [keras.layers.core.Dropout, 
                         keras.layers.core.Reshape,
                         keras.layers.core.Activation, 
                         keras.layers.normalization.BatchNormalization]
            self.display_layers = [l for l in self.nn.layers if not (type(l) in discarded)]
            self._define_activation_tf()
        self.interactive_display = interactive_display
        self.epoch_counter = 0
        self.type = _type
        self.mask=mask
        self.divN = {}
        if len(self.data('valid', 'y').shape) > 2:
            for key in ['train', 'valid', 'test']:
                d = self.data(key, 'y')
                self.divN[key] = 1/d.shape[0] * (d * self.mask).sum(axis=0).sum(axis=-1) * self.batch_size

        
    def get_param_values(self):
        params = []
        for layer in self.nn.layers:
            params += layer.get_weights()
        return params
    
    def params_monitor(self):
        for i, layer in enumerate(self.nn.layers):
            if len(layer.get_weights()) < 1:
                continue
            print(('layer (%d) params range: ' % (i+1)) + ' '.join(['[%.2f, %.2f]' % (p.min(), p.max()) for p in layer.get_weights()]))
        params = self.get_param_values()
        print('|params|^2 = %.2f, max |params| = %.2f' % 
              (np.sum([(p**2).sum() for p in params]), 
               np.max([np.max(np.abs(p)) for p in params])))
               
        
    def on_train_begin(self, logs={}):
        self.loss_history = []
        self.last_valid_loss = np.inf
        self.loss_history = {
            'train': [],
            'train ' + self.secondary: [],
            'valid': [],
            'valid ' + self.secondary: [],
            'test' : [],
            'test ' + self.secondary: [],
            'learning_rate': []
        }
        self.best_losses = {
            'valid': np.inf, 
            'valid ' + self.secondary: -np.inf,
            'test': np.inf, 
            'test ' + self.secondary: -np.inf, 
            'iter': 0,
            'epoch': 0,
            'params': [],
            'message': self.message
        }  
        
    def on_epoch_end(self, epoch, logs={}):
        self.last_logs = logs
        self.epoch_counter += 1
        self.loss_history['train'].append(logs['loss'])
        self.loss_history['valid'].append(logs['val_loss'])
        if 'acc' in logs:
            self.loss_history['train ' + self.secondary].append(logs['acc'])
            self.loss_history['valid ' + self.secondary].append(logs['val_acc'])
        elif ('acc_1') in logs:
            N = self.data('test', 'y').shape[1]
            for phase, phase_name in zip(['train', 'valid'], ['acc', 'val_acc']):
                accs = 1/self.divN[phase] * np.array([logs['%s_%d' % (phase_name, i + 1)] for i in range(N)])
                logs[phase_name + 's'] = accs
                self.loss_history[phase + ' ' + self.secondary].append(accs.mean())
         
        self.improvement.append(logs['val_loss'] < self.last_valid_loss)
        self.last_valid_loss = logs['val_loss']
        if self.interactive_display:
            plt.gca().cla()
            display.clear_output(wait=True)
        if self.interactive_display:
            self.plot()
            
        if logs['val_loss'] < self.best_losses['valid']:
            self.best_losses['valid'] = logs['val_loss']
            self.best_losses['valid ' + self.secondary] = logs['val_acc' + 's' * ('acc_1' in logs)]
            loss = self.nn.evaluate(self.data('test', 'X'), self.data('test', 'y'), 
                                    batch_size=self.batch_size)
            print('\nEpoch %d, test loss %f, test %s %f' % 
                  (self.epoch_counter, loss[0], self.secondary, loss[1]))
            self.best_losses['test'] = loss[0] 
            self.best_losses['test ' + self.secondary] = loss[1]                       
            self.best_losses['params'] = self.get_param_values()
            self.best_losses['iter'] = self.epoch_counter * self.data.end['train'] // self.batch_size
            self.best_losses['epoch'] = self.epoch_counter
#            with open(self.dump_file, 'wb') as f:
#                pickle.dump(self.best_losses, f)

        print(self.message)
        for phase, phase_name in zip(['train ', 'valid '], ['acc', 'val_acc']):
            print('\n%smean accuracy %.3f' % (phase, self.loss_history[phase + self.secondary][-1]))
            print(logs[phase_name + 's' * ('acc_1' in logs)])
        print('\n' + repr(self.params_monitor()))


    def plot(self):
        plt.gca().cla()
        plt.close()
        fig = plt.figure(1, (13, 9))
        axes_no = 2 + len(self.display_layers) * self.monitor_activations
        axes = [fig.add_subplot(int(np.ceil(axes_no/3)), 
                                2 + self.monitor_activations, 
                                i + 1) for i in np.arange(axes_no)]
        for ax, lossn in zip(axes[:2], ['', ' ' + self.secondary]):
            xx = np.arange(1, 1+len(self.loss_history['train'])) 
            ax.scatter(xx, self.loss_history['train' + lossn], label='train', marker='o')
            ax.scatter(xx, self.loss_history['valid' + lossn], label='valid', marker='x')
            ax.set_xlabel('epoch')
            ax.set_xticks(xx[::int(max(1, len(xx)/10, np.sqrt(len(xx)/2.5)))])
            ax.set_ylabel(('loss' + lossn).split(' ')[-1])
            ax.set_xlim(.5, len(xx) + .5)
            ax.set_title(('loss' + lossn).split(' ')[-1])

        if self.monitor_activations:
            sample = 160
            for ax, f, l in zip(axes[2:], self._activation_tf, self.display_layers):
#                act = np.concatenate([f(i) for i in np.arange(sample)])
                act = f([self.data('train', 'X')[:sample], 1]) # output in train mode = 1
                ax.imshow(act.transpose(), vmin=0, vmax=max(1, act.max()), 
                          aspect=sample/(l.output_shape[1] * 2),
                          interpolation='nearest')     
                ax.set_title(' '.join(repr(l).split(' ')[:2]))
                ax.grid('off')
        
#        fig.tight_layout()    
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc = (0.40, 0.02), ncol=2, fontsize=12)
        display.display(plt.gcf())

    def _define_activation_tf(self):
        self._activation_tf = []
        input = self.nn.layers[0].input
        for l in self.display_layers:
            self._activation_tf.append(K.function(
                [input, K.learning_phase()], 
                l.output
                ))

class ResetLSTM(keras.callbacks.Callback):
    """
    Class that resets keras model states on epoch begin.
    """
    def __init__(self, model, propagate=True):
        self.model = model
        self.propagate = propagate
        self.layerstates = None
        
    def on_epoch_end(self, epoch, logs={}):
        if self.propagate:
            self.layerstates = []
            for layer in self.model.layers:
                ls = []
                if hasattr(layer, 'states'):
                    for state in layer.states:
                        ls.append(state.eval())
                self.layerstates.append(ls)

    def on_epoch_begin(self, epoch, logs={}):
        if self.layerstates is None:
            return
        if self.propagate:
            for layer, saved_layer in zip(self.model.layers, self.layerstates):
                if hasattr(layer, 'states'):
                    for state, saved_state in zip(layer.states, saved_layer):
                        new_state = saved_state
                        new_state[1:, :] = new_state[:-1, :]
                        K.set_value(state, new_state)
        else:
            self.model.reset_states()

class PrintStates(keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.statelayers = []
        for layer in self.model.layers:
            if hasattr(layer, 'states'):
                self.statelayers.append(layer)
        
    def on_epoch_end(self, epoch, logs={}):
        display.clear_output(wait=True)
        plt.gca().cla()
        plt.close()   
        fig = plt.figure(1, (13, 9))
        axes_no = len(self.statelayers) * len(self.statelayers[0].states)
        axes = [fig.add_subplot(len(self.statelayers), 
                                len(self.statelayers[0].states), 
                                i + 1) for i in np.arange(axes_no)]
        for i, layer in enumerate(self.statelayers):
            for j, state in enumerate(layer.states):
                sh = state.eval().shape
                axes[i * len(layer.states) + j].imshow(state.eval(), interpolation='nearest', aspect=sh[1]/(2*sh[0]))
        display.display(plt.gcf())

def def_R2(meanSS):
    def R2(y_true, y_pred):
        errSS = T.mean((y_true - y_pred)**2)
        return 1 - errSS
    return R2
    

class LrReducer(keras.callbacks.Callback):
    """ 
    class to reduce learning rate in keras training procees.
    """
    def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1, monitor='val_loss', restore_best=True, reset_states=False):
        super(keras.callbacks.Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = np.inf if ('loss' in monitor) else -np.inf
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose
        self.monitor = monitor
        self.restore_best = restore_best
        self.saved_layers = None
        self.reset_states = reset_states
        
    def on_epoch_begin(self, epoch, logs={}):
        if self.verbose > 0:
            print('---current learning rate: %.8f' % K.get_value(self.model.optimizer.lr))
            
    def restore_params(self):
        if self.saved_layers is None:
            print('--- cannot restore params. No layers saved')
            return
        for layer, saved_layer in zip(self.model.layers, self.saved_layers):
            K.batch_set_value(list(zip(layer.weights, saved_layer)))
#            for weight, saved_weight in zip(layer.weights, saved_layer):
#                K.set_value(weight, saved_weight)
        print('--- best params restored. ')        
        
    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get(self.monitor)
        if (self.best_score - current_score) * (2*('loss' in self.monitor) - 1) > 0:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('\n---current best %s: %.5f' % (self.monitor, current_score))
            self.saved_layers = []
            for layer in self.model.layers:
#                    weights_to_save = []
#                    for weight in layer.weights:
#                        weights_to_save.append(K.get_value(weight))
#                    self.saved_layers.append(weights_to_save)
                self.saved_layers.append(K.batch_get_value(layer.weights))
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= self.reduce_nb:
                    self.wait = -1
                    lr = K.get_value(self.model.optimizer.lr)
                    K.set_value(self.model.optimizer.lr, np.float32(lr*self.reduce_rate))
                    if self.restore_best:
                        self.restore_params()
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1
        if self.reset_states:
            self.model.reset_states()
    
    def on_train_end(self, logs={}):
        self.restore_params()
            
class ThresholdStopper(keras.callbacks.Callback):
    def __init__(self, thresholds, monitor=['val_R2', 'R2'], dir=1, verbose=1):
        super(keras.callbacks.Callback, self).__init__()
        self.thresholds = thresholds
        self.verbose = verbose
        self.monitor = monitor
        self.dir = dir
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch <= 1:
            return
        if all([(logs.get(m) - t) * self.dir >= 0 for m, t in zip(self.monitor, self.thresholds)]):
            if self.verbose > 0:
                print("Epoch %d: threshold matched" % (epoch))
                for m, t in zip(self.monitor, self.thresholds):
                    sign = ('>=' if (self.dir > 0) else '<=')
                    print("%s: %.5f %s %.5f" % (m, logs.get(m), sign, t))
            self.model.stop_training = True
            
class TensorBoard(keras.callbacks.TensorBoard):
    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:

                for weight in layer.weights:
                    tf.summary.histogram(weight.name, weight)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = w_img.get_shape()
                        if len(shape) > 1 and shape[0] > shape[1]:
                            w_img = tf.transpose(w_img)
                        if len(shape) == 1:
                            w_img = tf.expand_dims(w_img, 0)
                        ### Here is the difference with the parent keras class
                        # w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        ###
                        if len(shape) < 3:
                            w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                        elif len(shape) == 3:
                            w_img = tf.expand_dims(w_img, -1)          
                        ###
                        tf.summary.image(weight.name, w_img)

                if hasattr(layer, 'output'):
                    tf.summary.histogram('{}_out'.format(layer.name),
                                         layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)    