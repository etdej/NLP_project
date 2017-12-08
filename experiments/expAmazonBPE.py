
# coding: utf-8

print('entered the file')

import sys
sys.path.insert(0,'/home/ecd353/NLP_project/')

import comet_ml
from comet_ml import Experiment
comet = Experiment(api_key="uQuKaohh924bv3c68Jhyumhw7", log_code=True)


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

all_hyp = {3: {'nb_merge': 50, 'l0': 852}, 4: {'nb_merge': 50, 'l0': 1176}, 5: {'nb_merge': 50, 'l0': 1176}, 6: {'nb_merge': 200, 'l0': 501}, 7: {'nb_merge': 200, 'l0': 771}, 8: {'nb_merge': 200, 'l0': 960}, 9: {'nb_merge': 500, 'l0': 1095}, 10: {'nb_merge': 500, 'l0': 501}, 11: {'nb_merge': 500, 'l0': 609}, 12: {'nb_merge': 1000, 'l0': 501}, 13: {'nb_merge': 1000, 'l0': 1230}, 14: {'nb_merge': 1000, 'l0': 1041}, 15: {'nb_merge': 2000, 'l0': 744}, 16: {'nb_merge': 2000, 'l0': 717}, 17: {'nb_merge': 2000, 'l0': 744}} 

exp_id = 26

nb_classes = 2
batch_size = 256

nb_merge = 100
l0 = 933
dropout_rate = 0.5
learning_rate = 0.0001

num_epochs = 15

save_model_path = '/home/ecd353/NLP_project/experiments/save/models/amazon/exp'+str(exp_id)+'_best.pth.tar'
save_pred_path = '/home/ecd353/NLP_project/experiments/save/predictions/amazon/exp'+str(exp_id)+'.txt'

hyper_params = {'learning_rate': learning_rate}
hyper_params.update(all_hyp[exp_id])

## generate dataset
print("generating dataset Amazon")
data_path='/home/ecd353/NLP_project/data/amazon/'
training_set, validation_set,list_subword_without_end,alphabet_size = dataGenerator(nb_merge=hyper_params['nb_merge'], file_name=data_path+'train.txt', max_length=hyper_params['l0'])

comet.log_multiple_params(hyper_params)
print("generating dataset done")


# Build, initialize model
model = Char_CNN_Small.Char_CNN_Small(hyper_params['l0'], alphabet_size, 0.5, nb_classes, batch_size)
model.init_weights()
model.cuda()

# Loss and Optimizer
loss = nn.BCELoss()
loss = loss.cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=hyper_params['learning_rate'])
print(model)

# Train the model
training_iter = data_iter(training_set, batch_size)
train_eval_iter = eval_iter(training_set[:256], batch_size)
validation_iter = eval_iter(validation_set, batch_size)

total_batches = int(len(training_set) / batch_size)

tls.training_loop(batch_size, total_batches, alphabet_size, hyper_params['l0'], num_epochs, model, loss, optimizer, 
              training_iter, validation_iter, train_eval_iter, save_model_path, comet, cuda=True)


# Loading best model and calculating accuracy on test set
tls.load_checkpoint(model, save_model_path)

test_set = dataGeneratorTest(list_subword_without_end,file_name=data_path+'test.txt', max_length=hyper_params['l0'])
test_iter = eval_iter(test_set, batch_size)
test_acc = tls.evaluate(model, test_iter, batch_size, alphabet_size, hyper_params['l0'], cuda=True, save_pred_file=save_pred_path)
print("Final test accuracy :  %f" %(test_acc))
print("Final test err :  %f" %( 1 - test_acc))
comet.log_accuracy(test_acc)



