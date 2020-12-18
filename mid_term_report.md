### Mid Term Review 

#### Overview

1. Literature Review
2. Main Paper
3. Three papers which build up on NBEATS
4. Three papers which inspired NBEATS
5. N-BEATS Code review
6. Direction for future work

------------------

#### 1. Literature Review

Neural networks based models for time series forecasting is gaining a lot of attention in academic research. Despite this, classical models like ARIMA are still widely used in practice. One major reason is that neural network models are seen as more complex, and require larger data sets to perform well. Classical models, on the other hand, have a reputation for being quite accurate and robust. Moreover, they are simple and easy to operate, making it attractive for non-expert users. Nevertheless, neural networks are growing more popular because they can model more general relationships in the data set, especially non-linearities. [1] 

Now, we comment on three common basic units of a neural network in the context of time series forecasting - multilayer perceptron (MLP), convolutional neural network (CNN) and recurrent neural network (RNN) [2]. When dealing with time series data, MLPs have a few shortcomings. Firstly, they have fixed number of inputs, and cannot deal with time series data of different lengths. They also don’t fully exploit the structure of time series data, that is they don’t interpret the input as sequential data. One possible suggestion is to explicitly give time in the input, so that MLPs adapt to the sequential nature of time series data. But, this has another problem- it cannot detect time invariant patterns very well. RNNs and CNNs are better-suited to time series forecasting than MLPs. Their architectures are designed to account for sequential nature of data- CNNs use filters that are shared across time steps while RNNs have information (as ‘hidden states’) of previous time steps fed to the future time steps. Since the dimensions of the weights matrices and filters used in RNNs and CNNs don’t depend on the length of input data, they can also handle time series data of variable lengths.

< N-BEATS has a MLP structure but with residual connections. Any similarities to RNN. Discuss this and relate to previous para. >

In our main paper [3], N-BEATS has a sequence-to-sequence model, that is input sequences are directly used to predict the output sequence for the entire future horizon. This is different from models that predict a single future time step and feed these predictions as inputs in the same models to predict the next future time steps. These models suffer from an accumulation of forecast errors, while N-BEATS and other sequence-to-sequence models don't. In this report, we have mainly reviewed sequence-to-sequence models. The next section summarises some such models that we found interesting from works done prior to [3].

------------------

#### 2. Three papers which inspired NBEATS
------------------

#### 3. Main Paper

------------------

#### 4. Three papers which build up on NBEATS

------------------

#### 5. N-BEATS Code review for m3 dataset

##### 5.1 Setting up and running N-BEATS model

The `MAKEFILE` in the repository targets a docker-based environment for building and testing N-BEATS. However, we decided to do it without Docker as it has it's own time and memory related issues. We built N-BEATS from the source code using basic environment variables and figured out how to run various experiments on Google Colab and on a Unix console. 

Demo for running an experiment on a bash terminal using N-BEATS for the 1.3Mb M3 dataset which is manually downloaded and placed in `storage/datasets/m3/`. 

```sh
git clone https://github.com/deeptavker/N-BEATS
cd N-BEATS
pip install -r requirements.txt
export PYTHONPATH=$PWD
export STORAGE=$PWD/storage
python datasets/m3.py M3Dataset download
python experiments/m3/main.py --config_path=$PWD/experiments/m3/generic.gin build_ensemble
python experiments/m3/main.py --config_path=$PWD/experiments/m3/interpretable.gin build_ensemble
python experiments/m3/main.py --config_path=storage/experiments/m3_generic/repeat=3,lookback=4,loss=MAPE/config.gin run
# GENERATED FILE for specific params : storage/experiments/m3_generic/repeat=3,lookback=4,loss=MAPE/forecast.csv
```
##### 5.2 Analysis of Results

##### 5.3 N-BEATS Code Structure & Review

```
.
└── N-BEATS
    ├── common
    ├── datasets
    ├── experiments
    ├── models
    ├── notebooks
    ├── storage
    ├── summary
    └── test
```

###### 5.3.1 common
```
├── common                                                                        
│   ├── __init__.py    
│   ├── experiment.py    
│   ├── http_utils.py     
│   ├── metrics.py     
│   ├── sampler.py   
│   ├── settings.py   
│   └── torch    
│       ├── __init__.py    
│       ├── losses.py     
│       ├── ops.py    
│       └── snapshots.py    
```

`experiment.py` 

This python module defines the base class for building experiment related configuration files based on ensemble parameters and also for specific combinations of *repeats*, *lookbacks* and *losses*. The *instance* method defined in this module is defined on a case to case basis for different experiments. This module is not intended for standalone execution, rather it is used as a supporting script for modules within the *experiments* directory and other custom logic. 

`settings.py`

This python module is used to configure directory paths for datasets, tests and experiment related generated data such as specific config files, snapshots and forecasts. Specifically, as a requirement, it takes in the environment variable named *STORAGE* as input from the environment. This env. var. must be set before proceeding as much of the remaining code for N-BEATS imports this particular module for path related queries. 

`metrics.py`

This module provides subroutines for commputing various metrics for `forecast` and `target` such as `MASE`, `MAPE`, `ND`, `NRMSE`, `SMAPE1`, `SMAPE2`. Reference for each is provided in the docstrings of respective functions. 

`http_utils.py`

This file provides subroutines for downloading files from the internet and saving them at a specific path. 

`sampler.py`




###### 5.3.2 datasets
```
├── datasets    
│   ├── __init__.py   
│   ├── m3.py   

```

`m3.py`

This python script is a helper module for downloading the m3 dataset and generating required `.npy` files for training and testing. It downloads the main dataset file `m3.xls` from `https://forecasters.org/data/m3comp/M3C.xls` and puts it in `storage/datasets/m3/`. Following the download, the script reads the dataset using a pandas dataframe and breaks it down into multiple `.npy` files viz. `horizons.npy`, `ids.npy`, `groups.npy`, `test.npy` and `training.npy`. The test-train split at this stage is dictated by the horizon value. The metadata for horizons is pre-loaded. There are 4 different seasonal patterns {year, quart, month, other} for which data is availble across various groups {macro, industry, finance, micro, demographic, others}. 

###### 5.3.2 experiments
```
├── experiments
│   ├── __init__.py
│   ├── m3
│   │   ├── __init__.py
│   │   ├── generic.gin
│   │   ├── interpretable.gin
│   │   └── main.py
│   ├── model.py
│   └── trainer.py
```

`m3/main.py`

This module implements the `instance` method for the `Experiment` class in `common/experiment.py`. In this implementation, this module uses modules from `common`, `experiment` and `summary` directories from `$PYTHONPATH` to create training sets and then trains those set on the model that is either `generic` or `interpretable`. It saves the forecast in a specific directory which is defined by the implementation of `Experiment` in `common/experiment.py`. It uses the `fire` library in order to take command line arguments which defines the `gin config` to be used and whether the script will **build ensemble config files** or **run a specific experiment** based on particular values of `loss`, `lookback` and `repeat`. This file also provides a subroutine to load the generated `.npy` files during an experiment. When used with an argument of `build_ensemble` it requires a `gin config` file in order to generate multiple config files in `storage/experiments/m3_generic` or `storage/experiments/m3_interpretable`. There config files reperesent various combinations of `loss`, `repeat` and `lookback` which can be run individually or in parallel. A `forecast` file for each combination of the aforementioned three parameters is generated in respective directories within `storage/experiments/`.  


###### 5.3.2 models
```
├── models
│   ├── __init__.py
│   └── nbeats.py
```
###### 5.3.2 notebooks
```
├── notebooks
│   ├── M3.ipynb
```

###### 5.3.2 storage
```
├── storage
│   ├── datasets
│   │   └── m3
│   │       ├── M3C.xls
│   │       ├── groups.npy
│   │       ├── horizons.npy
│   │       ├── ids.npy
│   │       ├── test.npy
│   │       └── training.npy
│   ├── experiments
│   └── test
```
###### 5.3.2 summary
```
├── summary
│   ├── __init__.py
│   ├── m3.py
│   └── utils.py
```
###### 5.3.2 test
```
└── test
    ├── __init__.py
    ├── __init__.pyc
    └── summary
        ├── __init__.py
        ├── __init__.pyc
        ├── test_m3.py
```


------------------

#### 6. Direction for future work

- Bayesian unertainty measures for detecting drifts in time series forecasts. 
- Modification of ensembling techniques (no training alteration) 

#### REFERENCES

[1] - H. Hewamalage, C. Bergmeir and K. Bandara, “Recurrent Neural Networks for Time Series Forecasting: Current status and future directions”, International Journal of Forecasting (2020).

[2] - K. Benidis, S.S. Rangapuram, V. Flunkert, B. Wang, D. Maddix, C. Turkmen, J Gasthaus, M. Bohlke-Schneider, D. Salinas, L. Stella, L. Callot, T. Januschowski, “Neural forecasting: Introduction and literature overview”, Preprint.

[3] - B.N. Oreshkin, D. Carpov, N. Chapados and Y. Bengio, "N-BEATS: Neural Basis Expansion Analysis for interpretable Time Series forecasting", ICLR (2020)
