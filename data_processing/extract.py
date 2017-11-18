alphabet = [' ','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v' ,'w', 'x', 'y', 'z', '0', '1', '2','3', '4', '5', '6', '7','8','9','-', ';', '.', '!', '?', ':',
            '\'', '\\', '|', '_', '@', '#', '$', '%', '\^', '&', '*','\'', '\~', '+', '-', '=', '<', '>','(', ')',
            '[',']', '{', '}']

indexing = { letter : i+1 for i, letter in enumerate(alphabet)}
indexing['UNK'] = len(alphabet)
indexing['No_letter'] = 0



import os
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from numpy import random
import torch

def dataGenerator(data_path, train_split=0.8,binary=False, max_length=1014):

    dataset = []

    for filename in ['pos', 'neg']:
        file_dir = join(data_path, 'aclImdb', 'train', filename)
        files = [f for f in listdir(file_dir) if isfile(join(file_dir, f))]
        for f in files:
            name = f.split('.')[0]
            name = name.split('_')
            id = name[0]
            rating = name[1]
            if(binary): 
                if(filename=='pos'): 
                    rating=1
                else: 
                    rating=0
            else:
                rating= int(rating) -1

            path = join(file_dir, f)
            review = torch.zeros(max_length).long()
            with open(path, encoding='utf-8') as myfile:
                data = myfile.read()
                for i in range(min(max_length,len(data))):
                    letter = data[i].lower()
                    if letter in alphabet:
                        review[i] = indexing[letter]
                    else:
                        review[i] = indexing['UNK']

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
    training_set, validation_set = dataGenerator()
    print("length training_set : ", len(training_set))
    print("length validation_set : ", len(validation_set))