
# coding: utf-8

print('entered the file')

import sys
sys.path.insert(0,'/home/rns365/NLP_project/')

import comet_ml
from comet_ml import Experiment
comet = Experiment(api_key="vNj6sy9OEjfNKjDOdJCbB5Gtl", log_code=True)


import torch.nn as nn
import torch
from data_processing.extractYelpBPE import *
from models import Char_CNN_Small
from models import tools as tls
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from numpy import random

nb_classes = 2
batch_size = 256

nb_merge = 100
l0 = 1014
dropout_rate = 0.5
learning_rate = 0.0001

num_epochs = 20
save_path = '/home/rns365/NLP_project/experiments/expBooksBPE_best.pth.tar'

## generate dataset
print("generating dataset")
data_path='/home/rns365/NLP_project/data/'
training_set, validation_set,list_subword_without_end,alphabet_size = dataGenerator(nb_merge=nb_merge, file_name=data_path+'trainYelp.txt', max_length=l0)

hyper_params = {'learning_rate': learning_rate, 
        'alphabet_size': alphabet_size, 
        'dropout_rate': dropout_rate,
        'l0': l0,
        'nb_merges': nb_merge}

comet.log_multiple_params(hyper_params)
print("generating dataset done")


# Build, initialize model
model = Char_CNN_Small.Char_CNN_Small(l0, alphabet_size, dropout_rate, nb_classes, batch_size)
model.init_weights()
model.cuda()

# Loss and Optimizer
loss = nn.BCELoss()
loss = loss.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
print(model)

# Train the model
training_iter = data_iter(training_set, batch_size)
train_eval_iter = eval_iter(training_set[:256], batch_size)
validation_iter = eval_iter(validation_set, batch_size)

total_batches = int(len(training_set) / batch_size)

tls.training_loop(batch_size, total_batches, alphabet_size, l0, num_epochs, model, loss, optimizer, 
              training_iter, validation_iter, train_eval_iter, save_path, comet, cuda=True)


# Loading best model and calculating accuracy on test set
tls.load_checkpoint(model, save_path)

test_set = dataGeneratorTest(list_subword_without_end,file_name=data_path+'test.txt', max_length=l0)
test_iter = eval_iter(test_set, batch_size)
test_acc = tls.evaluate(model, test_iter, batch_size, alphabet_size, l0, cuda=True)
print("Final test accuracy :  %f" %(test_acc))



