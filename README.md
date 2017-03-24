# nntimeseries

The repository provides the code for the paper 'Autoregressive Convolutional 
Neural Networks for Asynchronous Time Series' ([https://arxiv.org/abs/1703.04122](https://arxiv.org/abs/1703.04122))
submitted to ICML 2017), as well as general code for running grid serach on 
keras models. 

Files 'nnts/models/{CNN, LSTM, LR, SOCNN}.py' provide code for testing 
respective models, with the last one implementing the proposed 
Significance-Offset CNN.

Parameters for grid search can be specified in each of the above 
files. 

The repository supports optimization of the above models on artifical 
multivariate noisy AR time series and household electricity conspumption 
dataset
[https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
The dataset has to be specified alongside the paremeters in each of 
the files listed above. 

# Basic usage
1.(a) Generate artificial datasets by running 'python generate_artifical.py' or
  (b) Change the dataset to 'household.csv' in the appropriate model file.
2. Run any of the model scripts
   'python nnts/models/{CNN, LSTM, LR, SOCNN}.py'

Feel free to contact Mikolaj Binkowski ('mikbinkowski at gmail.com') with any 
questions and issues.

