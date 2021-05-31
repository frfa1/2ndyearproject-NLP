# Second Year Project - Group 4

#### Evaluating the effect of hand-crafted features in sentiment analysis

This repository contains all code in order to run the experiments conducted in the project *"Evaluating the affect of hand-crafted features in sentiment analysis"*

## Required data not in this repo

We are using GloVe word embeddings for the RNN model. The file containing these is too big to be in the repo, and one thus must add a folder named embeddings and add the [`glove.6B.50d.txt`](https://www.kaggle.com/watts2/glove6b50dtxt) file to it.

## How to reproduce our results

### Logistic regression

For the baseline logistic regression model, run the [`wrappers.py`](https://github.itu.dk/frph/2ndyearproject/blob/master/baseline_models/wrappers.py) module in baseline_models. The script contains wrapper functions for the baseline models such that various data inputs can easily be tested. 

Examples of results of these are in the same folder in [`results.txt`](https://github.itu.dk/frph/2ndyearproject/blob/master/baseline_models/results.txt)

Errors of the model when predicting on dev data can be seen in [`errors.csv`](https://github.itu.dk/frph/2ndyearproject/blob/master/baseline_models/errors.csv)

The wrapper module is curently set up to run the ablation study and prints it nicely to the console.

### RNN

For the RNN model, go to the models folder and run [`RNN.py`](https://github.itu.dk/frph/2ndyearproject/blob/master/models/RNN.py) - this module both contains the class definition of the model but also the wrapper function to run the model on some input data. 

Each of the `RNN_X.py` modules contain some variation in the included number of features for the ablation study. These are prepared for running on the ITU high performance cluster. Just ssh into hpc.itu.dk, copy the repo ***and*** embeddings onto the cluster and `sbatch` all the `runNN_X.sh` scripts.



