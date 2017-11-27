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

def dataGenerator2 (data_path= ' ', train_split=0.8,binary=False, max_length=1014, indexing=1):
    
    if(indexing==0):
        index = indexing
    if(indexing==1):
        index = bigIndexing
    if(indexing==2):
        index = altIndexing

    dataset = []
    dataloaded = np.loadtxt('Video_Games_5.txt',delimiter='\n', comments='\0',dtype=np.str)
    
    for rev in dataloaded: 
        print('ouloulou')
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
            
        #review = torch.zeros(max_length).long()
        
        review = np.zeros(max_length)
        
        for i in range(min(max_length,len(data))):
            letter = data[i].lower()
            if letter in alphabet:
                review[i] = index[letter]
            else:
                review[i] = index['UNK']

        dataset.append({'review': review, 'rating': rating})
        #dataset.append({'review': review, 'rating': torch.IntTensor([rating])})

    #random split 0.8 / 0.2
    dataset_train, dataset_val =  train_test_split(dataset, test_size=1-train_split)

    return dataset_train, dataset_val