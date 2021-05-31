# Second Year Project - Group 4

#### Evaluating the affect of hand-crafted features in sentiment analysis

This repository contains all code in order to run the experiments conducted in the project *"Evaluating the affect of hand-crafted features in sentiment analysis"*

## Required data not in this repo

We are using GloVe word embeddings for the RNN model. The file containing these is too big to be in the repo, and one thus must add a folder named embeddings and add the `glove.6B.50d.txt` file to it.

## How to reproduce our results

### Logistic regression

For the baseline logistic regression model, run the `wrappers.py` module in baseline_models. Wrapper functions and the call 
