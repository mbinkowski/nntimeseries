# nntimeseries

The repository provides code for running grid serach on keras models. 

Files CNN.py, LSTM.py and LR.py provide code for testing CNNs. LSTMs 
and Linear regression, respectively.
File CVI2.py provides code for testing proposed Significance-Offset 
Convolutional Neural Network.

Grid-search parameters can be specified in each of the above files. 

The repository supports optimization of the above models on artifical 
multivariate noisy AR time series and household electricity conspumption 
dataset
https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
The dataset has to be specified alongside the paremeters in each of 
the files listed above. 

Before running the optimization:
1. Specify the working directory in __init__ files
2. For artificial datasets, make sure the appropriate datasets have 
already been generated. To generate them, run generate_artifical.py.
3. Set the appropriate dataset names and parameter setting.


