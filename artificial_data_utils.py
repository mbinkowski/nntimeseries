# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:43:33 2017

@author: mbinkowski
"""

from __init__ import *
    
    
class BinaryNoise(object):
    def __init__(self, additive=True, scale=.05, p=None, random=False):
        self.additive = additive
        self.scale = scale
        self.offset = (np.random.rand(2) * 2 - 1)*scale if random else [-scale, scale]
        self.p = np.random.rand() if (p is None) else p
        
    def __call__(self, x):
        r = np.random.binomial(n=1, p=self.p, size=x.shape)
        adj = self.offset[1] * r + self.offset[0] * (1 - r)
        if self.additive:
            return x + adj
        else:
            return x * (1 + adj)

    def __repr__(self):
        ret = ('additive' if self.additive else 'multiplicative')
        return ret + (' Binary Noise, offsets = [%.5f, %.5f], p = %.2f' % (self.offset[0], self.offset[1], self.p))


class GaussianNoise(object):
    def __init__(self, additive=True, scale=.05):
        self.additive = additive
        self.scale = scale
    
    def __call__(self, x):
        adj = np.random.normal(size=x.shape) * self.scale
        if self.additive:
            return x + adj
        else:
            return x * (1 + adj)
    
    def __repr__(self):
        return ('additive' if self.additive else 'multiplicative') + (' Gaussian Noise, scale = %.5f' % self.scale)   
        
        
class NoisySignal(object):
    def __init__(self, n=10000, sources=2, exponential_time=False, single_source=True, 
                 order=10, e_sigma=.005, params_sum=.999, 
                 filepath=None, save=True):
        self.n = n
        self.sources = sources
        self.exponential_time = exponential_time
        self.single_source = single_source
        name = 'artificial' + self.__name__() + '.pickle'
        if (type(filepath) == bool) and (name in os.listdir('data/')):
            print('File already generated.')
            filepath = 'data/' + name
        if (filepath is None) or (type(filepath) == bool):
            print('Generating Noisy Signal...')
            self.order = order
            self.params_sum = params_sum
            self.e_sigma = e_sigma
            self._compute_signal()
            self._compute_noises()
            print('Done.')
            if save or filepath:
                self.save()
        else:
            print('Reading Noisy Signal from file ' + filepath + '...')
            with open(filepath, 'rb') as f:
                self.__dict__.update(pickle.load(f).__dict__)
            print('Done.')

        
    def _compute_signal(self):
        self.params = np.random.rand(self.order)
        self.params = self.params * self.params_sum/self.params.sum()
        x = np.random.normal(size=(self.order,)) * self.e_sigma
        if self.exponential_time:
            self.durations = np.ceil(np.random.exponential(scale=2, size=(self.n,))) + 1
        else:
            self.durations = np.ones(self.n)
        self.N = int(self.durations.sum()) + 1

        for i in range(self.N - self.order):
            new_x = (self.params * x[-self.order:]).sum() + np.random.normal(scale=self.e_sigma)
            x = np.concatenate([x, [new_x]])
        x = (x - x.mean())/x.std()
        self.x = x

    def _compute_noises(self):
        noises = []    
        for i in range(self.sources):
            scale = 2.0**((-i//4 - 4))
            if i%4 == 0:
                noises.append(BinaryNoise(additive=True, scale=scale))
            elif i%4 == 1:
                noises.append(BinaryNoise(additive=False, scale=scale))
            elif i%4 == 2:
                noises.append(GaussianNoise(additive=True, scale=scale))
            elif i%4 == 3:
                noises.append(GaussianNoise(additive=False, scale=scale))

        X = np.zeros((self.N, 1 + self.sources + self.single_source + 2*self.exponential_time))
        X[:, 0] = self.x

        if self.single_source:
            frequencies = 2.0**np.arange(self.sources)
            frequencies /= frequencies.sum()
            choice = np.random.choice(a=np.arange(self.sources), size=self.N, p=frequencies)
            for i, noise in enumerate(noises):
                ind = (choice == i) 
                X[ind, 1] = noise(self.x[ind])
                X[ind, i + 2] = 1
            self.names = ['original', 'noisy'] + ['source' + str(i) for i in np.arange(self.sources)]
        else:
            for i, noise in enumerate(noises):
                X[:, i + 1] = noise(self.x)
            self.names = ['original'] + ['source' + str(i) for i in np.arange(self.sources)]

        if self.exponential_time:
            self.names += ['valid', 'duration']
            d_ind = np.asarray(self.durations.cumsum(), dtype=np.int)
            X[d_ind, -1] = self.durations
            X[d_ind, -2] = 1
        self.noises = noises
        self.X = X
        self.df = pd.DataFrame(self.X, columns=self.names)
    
    def __repr__(self):
        return '\n'.join(['Noisy Signal'] + [noise.__repr__() for noise in self.noises])
       
    def __name__(self):
        return 'ET' + str(int(self.exponential_time)) + 'SS' + str(int(self.single_source)) + 'n' + str(int(self.n)) + 'S' + str(int(self.sources))
        
    def save(self, filepath='data/artificial'):
        filepath += self.__name__()
        print('Saving to .pickle and .csv')
        self.df.to_csv(filepath + '.csv')
        with open(filepath + '.pickle', 'wb') as f:
            pickle.dump(self, f)
            
    def __call__(self):
        return self.df


class ArtificialGenerator(Generator):
    def __init__(self, filename='data/artificialET0SS0n10000S2.csv', 
                 train_share=(.8, 1.), input_length=1, output_length=1, verbose=1, 
                 limit=np.inf, batch_size=16, diffs=False):
        self.filename = filename
        X = pd.read_csv(filename, index_col=0)
        super(ArtificialGenerator, self).__init__(X, train_share=train_share, 
                                                  input_length=input_length, 
                                                  output_length=output_length, 
                                                  verbose=verbose, limit=limit,
                                                  batch_size=batch_size,
                                                  diffs=diffs)  
    
    def make_io_func(self, io_form, cols=[0], input_cols=None):
        if input_cols is None:
            input_cols = np.arange(1, len(self.X.columns))
        return super(ArtificialGenerator, self).make_io_func(io_form, cols=cols, 
                                                             input_cols=input_cols)
    def get_dim(self):
        return super(ArtificialGenerator, self).get_dim() - 1