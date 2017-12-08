
# coding: utf-8

# In[279]:
print('entered the file')

import sys
sys.path.insert(0,'/home/ecd353/NLP_project/')

import comet_ml
from comet_ml import Experiment
comet = Experiment(api_key="uQuKaohh924bv3c68Jhyumhw7", log_code=True)

# In[421]:

import torch.nn as nn
import torch
from data_processing.extractYelp import *
from models import Char_CNN_Small
from models import tools as tls
import numpy as np

import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from numpy import random

# Hyper Parameters

exp_index = -2

dict_hyper_params = {-2: {'dropout': 0.5119303689409218, 'l0': 1338}, -1:{'dropout': 0.5, 'l0': 1014}, 0: {'dropout': 0.45, 'l0': 636}, 1: {'dropout': 0.33, 'l0': 663}, 2: {'dropout': 0.66, 'l0': 528}} 

hyper_params = {'learning_rate': 0.0001, 
		'alphabet_size': 94}
hyper_params.update(dict_hyper_params[exp_index])


comet.log_multiple_params(hyper_params)

nb_classes = 2
batch_size = 128

num_epochs = 10
save_model_path = '/home/ecd353/NLP_project/experiments/save/models/exp'+str(exp_index)+'_best.pth.tar'
save_pred_path = '/home/ecd353/NLP_project/experiments/save/predictions/exp'+str(exp_index)+'.txt'

## generate dataset
print("generating dataset")
data_path='/home/ecd353/NLP_project/data/yelp/'
training_set, validation_set = dataGenerator(data_path+'train.txt', max_length=hyper_params['l0'])

print("generating dataset done")

# Build, initialize model
model = Char_CNN_Small.Char_CNN_Small(hyper_params['l0'], hyper_params['alphabet_size'], hyper_params['dropout'], nb_classes, batch_size)
model.init_weights()
model.cuda()

# Loss and Optimizer
loss = nn.BCELoss()
loss = loss.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=hyper_params['learning_rate'])

# In[438]:

print(model)


# In[439]:

# Train the model
training_iter = data_iter(training_set, batch_size)
train_eval_iter = eval_iter(training_set[:256], batch_size)
validation_iter = eval_iter(validation_set, batch_size)

total_batches = int(len(training_set) / batch_size)
#total_batches = 100


# In[441]:
tls.training_loop(batch_size, total_batches, hyper_params['alphabet_size'], hyper_params['l0'], num_epochs, model, loss, optimizer, 
              training_iter, validation_iter, train_eval_iter, save_model_path, comet, cuda=True)


# Loading best model and calculating accuracy on test set
tls.load_checkpoint(model, save_model_path)

test_set = dataGenerator(data_path+'test.txt', test=True, max_length=hyper_params['l0'])
test_iter = eval_iter(test_set, batch_size)
test_acc = tls.evaluate(model, test_iter, batch_size, hyper_params['alphabet_size'], hyper_params['l0'], cuda=True, save_pred_file=save_pred_path)
print("Final test accuracy :  %f" %(test_acc))




# In[ ]:



