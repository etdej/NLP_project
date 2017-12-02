#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:31:49 2017

@author: saad
"""

import numpy as np 
import os
import json
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from numpy import random
import torch
import re
from langdetect import detect
import random


alphabet = [' ','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v' ,'w', 'x', 'y', 'z', '0', '1', '2','3', '4', '5', '6', '7','8','9','-', ';', '.', '!', '?', ':',
            '\'', '\\', '|', '_', '@', '#', '$', '%', '\^', '&', '*','\'', '\~', '+', '-', '=', '<', '>','(', ')',
            '[',']', '{', '}','"']

indexing = { letter : i+1 for i, letter in enumerate(alphabet)}
indexing['UNK'] = len(alphabet)+1
indexing['No_letter'] = 0
bigAlphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
p=len(indexing)
bigIndexing = { letter : i+2 for i,letter in enumerate(bigAlphabet)}
altIndexing = { letter : i+p+1 for i,letter in enumerate(bigAlphabet)}
bigIndexing.update(indexing)
altIndexing.update(indexing)


def shuffle(): 
    with open('review.txt','r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open('review_shuffle.txt','w') as target:
        for _, line in data:
            target.write( line )

def createFiles(file_name='review_shuffle.txt', batch_size=0,total_size=398000,train_proportion=360/398): 
    file_object = open(file_name,"r")
    counter =0
    counter_pos=0
    counter_neg=0
    train_size=(int)(train_proportion*total_size)
    test_size= total_size - train_size
    print(test_size)
    print(train_size)
    
    file_train=open("train.txt","w")
    file_test=open("test.txt","w")

    with open(file_name) as infile:
        while(counter<train_size):
            text = infile.readline()
            
            if len(text)==0: 
                print('break')
                break
            
            
            info=json.loads(text)
            rating = info['stars']
            review = info['text']
            item={}
            item['review']=review
            binary=int(rating>3)
            item['rating']=binary
            
            
            try:
                language = detect(review)
                
            except: 
                print('failure to detect language')
                continue
            
            if(language!='en'):
                print(detect(review))
                continue
            
            if(binary==0 and counter_neg<train_size/2):
                json.dump(item,file_train)
                file_train.write('\n')
                counter_neg+=1
                counter+=1
                print(counter)
                
            elif(binary==1 and counter_pos < train_size/2):
                json.dump(item,file_train)
                file_train.write('\n')
                counter_pos+=1
                counter+=1
                print(counter)
                
        file_train.close()       
        counter=0
        counter_neg=0
        counter_pos=0
        
        while(counter<test_size):
            text = infile.readline()
            if len(text)==0: 
                print('break')
                break
            info=json.loads(text)
            rating = info["stars"]
            review = info["text"]
            item={}
            item["review"]=review
            binary=int(rating<4)
            item["rating"]=binary
            
            if(detect(review)!='en'):
                print(detect(review))
                continue
            
            if(binary==0 and counter_neg<test_size/2):
                json.dump(item,file_test)
                file_test.write('\n')
                counter_neg+=1
                counter+=1
                
            if(binary==1 and counter_pos < test_size/2):
                json.dump(item,file_test)
                file_test.write('\n')
                counter_pos+=1
                counter+=1
                
                
        file_test.close()
                
                
                
def dataGenerator(file_name, test=False,train_split=0.8, max_length=1014, indexing_choice=0):
    
    if(indexing_choice==0):
        index = indexing
    if(indexing_choice==1):
        index = bigIndexing
    if(indexing_choice==2):
        index = altIndexing
        
    counter=0
    dataset = []
    with open(file_name) as infile:
        while True: 
            text=infile.readline() 
            counter+=1
            if len(text)==0:
                break
            
            info=json.loads(text)
            rating = info["rating"]
            data = info["review"]
            
        
            review = torch.zeros(max_length).long()
            
            for i in range(min(max_length,len(data))):
                letter = data[i].lower()
                
                if letter in alphabet:
                    review[i] = index[letter]
                else:
                    review[i] = index['UNK']
        
            dataset.append({'review': review, 'rating':torch.IntTensor([rating])})
            
    #random split 0.8 / 0.2
    if test:
        return dataset
    else  :
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
    #shuffle()
    #createFiles()
    training_set, validation_set = dataGenerator('train.txt')
    test_set = dataGenerator('test.txt',test=True)
    print("length training_set : ", len(training_set))
    print("length validation_set : ", len(validation_set))
    print("length test_set : ", len(test_set))
                
        