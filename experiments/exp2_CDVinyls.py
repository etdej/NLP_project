
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
from data_processing.extract2 import *
from models import Char_CNN_Small
from models import tools as tls
import numpy as np

import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from numpy import random

# In[423]:

alphabet = [' ','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v' ,'w', 'x', 'y', 'z', '0', '1', '2','3', '4', '5', '6', '7','8','9','-', ';', '.', '!', '?', ':',
            '\'', '\\', '|', '_', '@', '#', '$', '%', '\^', '&', '*','\'', '\~', '+', '-', '=', '<', '>','(', ')',
            '[',']', '{', '}']

indexing = { letter : i+1 for i, letter in enumerate(alphabet)}
indexing['UNK'] = len(alphabet)
indexing['No_letter'] = 0




# In[437]:

# Hyper Parameters


hyper_params = {'learning_rate': 0.0001, 
		'alphabet_size': 94, 
		'dropout_rate': 0.5,
		'l0': 1014 }


comet.log_multiple_params(hyper_params)

nb_classes = 2
batch_size = 128

num_epochs = 20
save_path = '/home/ecd353/NLP_project/experiments/exp2_best.pth.tar'

## generate dataset
print("generating dataset")
data_path='/home/ecd353/NLP_project/data/'
training_set, validation_set = dataGenerator(data_path+'train.txt', max_length=hyper_params['l0'])

print("generating dataset done")

# Build, initialize model
model = Char_CNN_Small.Char_CNN_Small(hyper_params['l0'], hyper_params['alphabet_size'], hyper_params['dropout_rate'], nb_classes, batch_size)
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


tls.training_loop(batch_size, total_batches, alphabet_size, l0, num_epochs, model, loss, optimizer, 
              training_iter, validation_iter, train_eval_iter, save_path, comet, cuda=True)


# Loading best model and calculating accuracy on test set
tls.load_checkpoint(model, save_path)

test_set = dataGeneratorTest(file_name=data_path+'test.txt', max_length=hyper_params['l0'])
test_iter = test_iter(test_set, batch_size)
test = tls.evaluate(model, test_iter, batch_size, hyper_params['alphabet_size'], hyper_params['l0'], cuda=True)
print("Final test accuracy :  %f" %(test_acc))




# In[ ]:



