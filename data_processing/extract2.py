#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 17:22:59 2017

@author: saad
"""

import numpy as np 
import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from numpy import random
import torch
import re


alphabet = [' ','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v' ,'w', 'x', 'y', 'z', '0', '1', '2','3', '4', '5', '6', '7','8','9','-', ';', '.', '!', '?', ':',
            '\'', '\\', '|', '_', '@', '#', '$', '%', '\^', '&', '*','\'', '\~', '+', '-', '=', '<', '>','(', ')',
            '[',']', '{', '}','"']

indexing = { letter : i+1 for i, letter in enumerate(alphabet)}
indexing['UNK'] = len(alphabet)
indexing['No_letter'] = 0
bigAlphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
p=len(indexing)
bigIndexing = { letter : i+2 for i,letter in enumerate(bigAlphabet)}
altIndexing = { letter : i+p+1 for i,letter in enumerate(bigAlphabet)}
bigIndexing.update(indexing)
altIndexing.update(indexing)

def dataGenerator2 (data_path= ' ', train_split=0.8,binary=False, max_length=1014, indexing_choice=0):
    
    if(indexing_choice==0):
        index = indexing
    if(indexing_choice==1):
        index = bigIndexing
    if(indexing_choice==2):
        index = altIndexing

    dataset = []
    dataloaded = np.loadtxt('first100lines.txt',delimiter='\n', comments='\0',dtype=np.str)
    for rev in dataloaded: 
        
        liste = re.split('"reviewText": ',rev)
        text= re.split(', "overall": ',liste[1])
        data = text[0]
        rating = float(re.split(',',text[1])[0])
        
        if(binary):
            if(rating>2):
                rating=1
            else:
                rating=0
        else:
            rating= int(rating) -1
            
        review = torch.zeros(max_length).long()
        
        #review = np.zeros(max_length)
        
        for i in range(min(max_length,len(data))):
            letter = data[i].lower()
            
            if letter in alphabet:
                review[i] = index[letter]
            else:
                review[i] = index['UNK']

        dataset.append({'review': review, 'rating': torch.IntTensor([rating])})
    #random split 0.8 / 0.2
    dataset_train, dataset_val =  train_test_split(dataset, test_size=1-train_split)

    return dataset_train, dataset_val



def data_iter(dataset, batch_size=32):
    dataset_size = len(dataset)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)
        batch_indices = order[start:start + batch_size]
        yield [dataset[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model.
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue

    return batches

if __name__ == '__main__':
    training_set, validation_set = dataGenerator2()
    print("length training_set : ", len(training_set))
    print("length validation_set : ", len(validation_set))
