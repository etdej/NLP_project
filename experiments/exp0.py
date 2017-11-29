import torch.nn as nn

from data_processing.extract2 import *
from models.char_CNN_Small import *
from models.tools import *

## generate dataset
print("generating dataset")
training_set, validation_set = dataGenerator()


# Hyper Parameters


fully_layers = [1024, 1024]
l0 = 1014
alphabet_size = 68
nb_classes = 10
batch_size = 8

learning_rate = 0.008
num_epochs = 20


# Build, initialize model
model = Char_CNN_Small(fully_layers, l0, alphabet_size, nb_classes, batch_size)
model.init_weights()

# Loss and Optimizer

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
training_iter = data_iter(training_set[:8], batch_size)
train_eval_iter = eval_iter(training_set[:8], batch_size)
validation_iter = eval_iter(validation_set[:8], batch_size)

training_loop(batch_size, alphabet_size, l0, num_epochs, model, loss, optimizer, training_iter, validation_iter, train_eval_iter)