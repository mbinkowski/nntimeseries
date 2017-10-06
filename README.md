# nntimeseries

The repository provides the code for the paper  [*Autoregressive Convolutional 
Neural Networks for Asynchronous Time Series*](https://arxiv.org/abs/1703.04122), as well as general code for running grid serach on keras models. 

Files 'nnts/models/{CNN, LSTM, LSTM2, LR, SOCNN}.py' provide code for testing 
respective models, with the last one implementing the proposed 
Significance-Offset CNN and LSTM2 implementing multi-layer LSTM.

**Basic Usage**

Each of the model files can be run as a script, e.g.
- `python ./CNN.py --dataset=artificial`   ### default save file 
-	`python ./SOCNN.py --dataset=household --save_file=results\\household_0.pkl`

Parameters for grid search can be specified in each of the above 
files. 

Each of these files defines a model class that can be imported and used on external dataset, as shown in example.ipynb file.

The repository supports optimization of the above models on artifical 
multivariate noisy AR time series and household electricity conspumption 
dataset
[https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
The dataset has to be specified alongside the paremeters in each of 
the files listed above. 

To generate aritficial datasets used in model evaluation in the paper, run 'python generate_artifical.py'.

**Requirements**
- python   >= 3.5.3
- Keras    >= 2.0.2
- numpy    >= 1.12.12
- pandas   >= 0.19.2
- h5py     >= 2.6.0

Feel free to contact Mikolaj Binkowski ('mikbinkowski at gmail.com') with any 
questions and issues.
