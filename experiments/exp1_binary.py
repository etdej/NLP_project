# coding: utf-8


import sys
sys.path.insert(0,'/Users/rubenstern/Desktop/NYU/Fall2017/NLP/NLP_project/')
import comet_ml
from comet_ml import Experiment

import torch.nn as nn
import torch
from data_processing.extract2 import *
from models import Char_CNN_Small
from models import tools as tls
import numpy as np


import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from numpy import random

comet = Experiment(api_key="vNj6sy9OEjfNKjDOdJCbB5Gtl", log_code=True)

## generate dataset
print("generating dataset")
data_path='/Users/rubenstern/Desktop/NYU/Fall2017/NLP/NLP_project/data_processing/'
training_set, validation_set = dataGenerator2(data_path, binary=True)
print("generating dataset done")


# Hyper Parameters

fully_layers = [1024, 1024]
l0 = 1014
alphabet_size = 94
nb_classes = 2
batch_size = 256

learning_rate = 0.001
num_epochs = 30

# Build, initialize model
model = Char_CNN_Small.Char_CNN_Small(l0, alphabet_size, nb_classes, batch_size)
model.init_weights()

# Loss and Optimizer
loss = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())


# Train the model
training_iter = data_iter(training_set, batch_size)
train_eval_iter = eval_iter(training_set[:500], batch_size)
validation_iter = eval_iter(validation_set, batch_size)

total_batches = int(len(training_set) / batch_size)

tls.training_loop(batch_size, total_batches, alphabet_size, l0, num_epochs, model, loss, optimizer, 
              training_iter, validation_iter, train_eval_iter,comet)




