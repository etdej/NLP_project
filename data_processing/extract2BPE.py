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
import re, collections
from nltk import word_tokenize, RegexpTokenizer


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


def create_initial_vocab(list_text):
    tokenizer = RegexpTokenizer(r'\w+')
    counter = collections.Counter()
    for text in list_text:
        test_tok = tokenizer.tokenize(text.lower())
        counter.update(test_tok)
    counter_dict = dict(counter)
    vocab = {}
    for key in counter_dict.keys():
        new_key = ' '.join(list(key)) + ' </w>'
        vocab[new_key] = counter_dict[key]
    return vocab    
    
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def create_vocab(list_text,num_merges):  
    print("start training BPE")
    list_subword = []
    vocab = create_initial_vocab(list_text)
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        list_subword.append(best[0]+best[1])
        vocab = merge_vocab(best, vocab)
    return list_subword    

def create_list_sobword_without_end(list_subword):
    list_subword_witout_end = []
    for i in list_subword:
        if len(i)>4 and i[-4:]=='</w>':
            if len(i[:-4])>1:
                list_subword_witout_end.append(i[:-4])
        else:
            list_subword_witout_end.append(i)
    return list_subword_witout_end

def transform_BPE_word(word,list_subwords):
    res = []
    current = ''
    for char in word:
        current += char
        in_one_sub = False
        list_concerned_sub = []
        list_len_concerned_sub = []
        max_len_concerned_sub = 0
        for sub in list_subwords:
            if current in sub and sub[:len(current)]==current:
                in_one_sub = True
                list_concerned_sub.append(sub)
                list_len_concerned_sub.append(len(sub))
                max_len_concerned_sub = max(max_len_concerned_sub,len(sub))
        if len(current)==max_len_concerned_sub:
            res.append(current)
            current = ''
        if in_one_sub==False:
            if len(current)==1:
                res.append(current)
                current = ''
            else:
                res.append(current[:-1])
                current = current[-1]
    if current != '':
        res.append(current)
    return res

def dataGenerator(file_name='../data/train.txt',train_split=0.8,binary=True, max_length=1014, indexing_choice=0,nb_merge=50):    
    if(indexing_choice==0):
        index = indexing
    if(indexing_choice==1):
        index = bigIndexing
    if(indexing_choice==2):
        index = altIndexing  
    list_string = []
    with open(file_name) as infile:
        while True: 
            rev=infile.readline() 
            if len(rev)==0:
                break
            liste = re.split('ratingInfo: ',rev)
            data = re.split('reviewInfo: ',liste[0])
            data=data[1]
            list_string.append(data)
    list_subword = create_vocab(list_string,nb_merge)
    print("end training BPE")
    list_subword_witout_end = create_list_sobword_without_end(list_subword)
    start_index = max(index.values())
    for i,sub in enumerate(list_subword_witout_end):
        index[sub] = start_index+i+1
    print('index',index)

    dataset = []
    print("start encoding training")
    with open(file_name) as infile:
        while True: 
            rev=infile.readline() 
            if len(rev)==0:
                break
            liste = re.split('ratingInfo: ',rev)
            data = re.split('reviewInfo: ',liste[0])
            data=data[1]
            rating = float(liste[1])
            if(binary):
                if(rating<4):
                    rating=0
                else:
                    rating=1
            else:
                rating= int(rating) -1
                
            review = torch.zeros(max_length).long()
            tokenizer = RegexpTokenizer(r'\w+')
            list_words = tokenizer.tokenize(data)
            list_word_subwords = [transform_BPE_word(i,list_subword_witout_end) for i in list_words]
            list_subwords = []
            for word in list_word_subwords:
                list_subwords += word
                list_subwords += ' '
            
            for i in range(min(max_length,len(list_subwords))):
                unit = list_subwords[i].lower()
                if unit in index:
                    review[i] = index[unit]
                else:
                    review[i] = index['UNK']
            dataset.append({'review': review, 'rating':torch.IntTensor([rating])})
    print("end encoding training")
    #random split 0.8 / 0.2
    dataset_train, dataset_val =  train_test_split(dataset, test_size=1-train_split)
    alphabet_size = max(index.values())+1
    return dataset_train, dataset_val,list_subword_witout_end,alphabet_size

def dataGeneratorTest(list_subword_without_end,file_name='../data/test.txt', binary=True, max_length=1014, indexing_choice=0):  
    if(indexing_choice==0):
        index = indexing
    if(indexing_choice==1):
        index = bigIndexing
    if(indexing_choice==2):
        index = altIndexing        
    dataset = []
    print("start encoding test")
    with open(file_name) as infile:
        while True: 
            rev=infile.readline() 
            if len(rev)==0:
                break               
            liste = re.split('ratingInfo: ',rev)
            data = re.split('reviewInfo: ',liste[0])  
            data=data[1]
            rating = float(liste[1])
            if(binary):
                if(rating<4):
                    rating=0
                else:
                    rating=1
            else:
                rating= int(rating) -1 
            review = torch.zeros(max_length).long()
            tokenizer = RegexpTokenizer(r'\w+')
            list_words = tokenizer.tokenize(data)
            list_word_subwords = [transform_BPE_word(i,list_subword_without_end) for i in list_words]
            list_subwords = []
            for word in list_word_subwords:
                list_subwords += word
                list_subwords += ' '

            for i in range(min(max_length,len(list_subwords))):
                unit = list_subwords[i].lower()
                if unit in index:
                    review[i] = index[unit]
                else:
                    review[i] = index['UNK']
            dataset.append({'review': review, 'rating':torch.IntTensor([rating])}) 
    print("end encoding test")
    return dataset



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
    #createFiles()
    training_set, validation_set,list_subword_without_end,alphabet_size = dataGenerator(nb_merge=300,file_name='../data/test.txt')
    test_set = dataGeneratorTest(list_subword_without_end)
    print("length training_set : ", len(training_set))
    print("length validation_set : ", len(validation_set))
    print("length test_set : ", len(test_set))
    
