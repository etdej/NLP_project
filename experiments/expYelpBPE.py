
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

all_hyp = {3: {'nb_merge': 50, 'dropout': 0.33130377077729595, 'l0': 1068}, 4: {'nb_merge': 50, 'dropout': 0.35509255409961304, 'l0': 609}, 5: {'nb_merge': 50, 'dropout': 0.4433443295386998, 'l0': 1014}, 6: {'nb_merge': 200, 'dropout': 0.5495640125810242, 'l0': 825}, 7: {'nb_merge': 200, 'dropout': 0.685203406754094, 'l0': 906}, 8: {'nb_merge': 200, 'dropout': 0.6268521197666834, 'l0': 906}, 9: {'nb_merge': 500, 'dropout': 0.6318302646662899, 'l0': 636}, 10: {'nb_merge': 500, 'dropout': 0.4723036947235081, 'l0': 1176}, 11: {'nb_merge': 500, 'dropout': 0.6391078766515466, 'l0': 447}, 12: {'nb_merge': 1000, 'dropout': 0.5766438239880094, 'l0': 1041}, 13: {'nb_merge': 1000, 'dropout': 0.611180497184276, 'l0': 1230}, 14: {'nb_merge': 1000, 'dropout': 0.47030607065598456, 'l0': 690}, 15: {'nb_merge': 2000, 'dropout': 0.6682737718497871, 'l0': 1257}, 16: {'nb_merge': 2000, 'dropout': 0.5211428669676292, 'l0': 1176}, 17: {'nb_merge': 2000, 'dropout': 0.3174179403870619, 'l0': 771},
19: {'nb_merge': 50, 'dropout': 0.5809063513934272, 'l0': 1095}, 20: {'nb_merge': 50, 'dropout': 0.4966674646230081, 'l0': 1095}, 21: {'nb_merge': 200, 'dropout': 0.5126193889939421, 'l0': 1122}, 22: {'nb_merge': 200, 'dropout': 0.5475755839973667, 'l0': 1068}, 23: {'nb_merge': 500, 'dropout': 0.5964290748209726, 'l0': 1122}, 24: {'nb_merge': 500, 'dropout': 0.44057388813981546, 'l0': 1095}, 25: {'nb_merge': 1000, 'dropout': 0.4491760085840652, 'l0': 960}, 26: {'nb_merge': 1000, 'dropout': 0.41483379077758464, 'l0': 1068}, 27: {'nb_merge': 2000, 'dropout': 0.42372442100806584, 'l0': 933}, 28: {'nb_merge': 2000, 'dropout': 0.5575083739757174, 'l0': 1149}}

exp_id = 24

nb_classes = 2
batch_size = 256

nb_merge = 100
l0 = 933
dropout_rate = 0.5
learning_rate = 0.0001

num_epochs = 15

save_model_path = '/home/ecd353/NLP_project/experiments/save/models/exp'+str(exp_id)+'_best.pth.tar'
save_pred_path = '/home/ecd353/NLP_project/experiments/save/predictions/exp'+str(exp_id)+'.txt'

hyper_params = {'learning_rate': learning_rate}
hyper_params.update(all_hyp[exp_id])

## generate dataset
print("generating dataset Yelp")
data_path='/home/ecd353/NLP_project/data/yelp/'
training_set, validation_set,list_subword_without_end,alphabet_size = dataGenerator(nb_merge=hyper_params['nb_merge'], file_name=data_path+'train.txt', max_length=hyper_params['l0'])

comet.log_multiple_params(hyper_params)
print("generating dataset done")


# Build, initialize model
model = Char_CNN_Small.Char_CNN_Small(hyper_params['l0'], alphabet_size, hyper_params['dropout'], nb_classes, batch_size)
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



