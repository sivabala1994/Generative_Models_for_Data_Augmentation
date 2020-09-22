#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:30:14 2019

@author: siv.bala
"""
from keras.utils.np_utils import to_categorical
import numpy as np
from PIL import Image
import random

#function to get the labels for a classification problem
def get_classes(dataset_name,split):
    directory="/Users/siv.bala/Desktop/keras_data/datasets/"+dataset_name+"/"+split+"/"
    lines=[]
    with open(directory+"labels.txt",'r') as f:
        lines=f.read().split("\n")
#    return lines[:-1]
    Y=[]
    for item in lines[:-1]:
        Y.append(item.split(" ")[0])
        
    return Y
        
    
    



lines=get_classes('cifar10','train')   
#function to get number of classes
def get_numclasses(dataset_name, split):
    Y=get_classes(dataset_name,split)
    num=len(set(Y))
    return num

numclasses=get_numclasses('fashionmnist','train')
#function to get number of samples
def num_samples(dataset_name,split):
    Y=get_classes(dataset_name,split)
    return len(Y)

numsamples=num_samples('fashionmnist','train')

#function to get number of examples per class
def num_samples_per_class(dataset_name,split):
    Y=get_classes(dataset_name,split)
    num=len(Y)
    unique, counts = np.unique(Y, return_counts=True)
    percent=np.array(counts)
    percentage=100*np.true_divide(percent,num)
    egs_percent=dict(zip(unique,percentage))
    
    egs_count=dict(zip(unique, counts))
    return egs_count,egs_percent


nspc,pernspc=num_samples_per_class('fashionmnist','train')


#function to get small datasets into numpy array structure
def get_data(dataset_name,split,imgnum):
    directory="/Users/siv.bala/Desktop/keras_data/datasets/"+dataset_name+"/"+split+"/"
    img = Image.open(directory+"img_"+str(imgnum)+".jpeg")
    img.load()
    data = np.asarray( img, dtype="int32" )
    
    return data
#X=get_data('fashionmnist','train',2)
    
#gives shape of the numpy array for the dataset
#def get_arrshape(dataset_name,split):
#    X=get_data(dataset_name,split)
#    return X.shape

def get_numchannels(dataset_name,split):
    return len(get_data(dataset_name,split,1).shape)

numchannels=get_numchannels('cifar10','train')

def get_imshape(dataset_name,split):
    return get_data(dataset_name,split,1).shape
imshape=get_imshape("cifar10","train")


def plot_img(array):
    if len(array.shape)==3:
        img = Image.fromarray(array, 'RGB')
        
    else:
        img = Image.fromarray(array, 'L')
        
    img.show()
    
def plot_n_random_img(dataset_name,split,n):
    
    y=num_samples(dataset_name,split)
    r=random.sample(range(0, y), n)
    for item in r:
        plot_img(get_data(dataset_name,split,item))


#y,y_per=num_egs_per_class('fashionmnist','test')


#X=get_data('cifar10','test')



#plot_n_random_img('mnist','train',3)


#%%

#dataloader for image dataset
directory="/Users/siv.bala/Desktop/keras_data/datasets/cifar10/train/labels.txt"
def generate_data(directory, batch_size):
    
    i = 0
    lines=[]
    with open(directory,'r') as f:
        lines=f.read().split("\n")
        
    imagelist=[]
    labellist=[]
    for item in lines[:-1]:
        imagelist.append((directory.replace("labels.txt","")+item.split(" ")[1]))
        labellist.append(int(item.split(" ")[0]))
    dic=dict(zip(imagelist, labellist))
#    random.shuffle(imagelist)
    while True:
        image_batch = []
        labels=[]
        for b in range(batch_size):
            if i == len(imagelist):
                i = 0
                random.shuffle(imagelist)
            sample = imagelist[i]
            
            
            i += 1
            pic = Image.open(sample)
            image = np.asarray(Image.open(sample))
            
            image_batch.append(image.astype("float32")/255.0)
            labels.append(dic[sample])
    
    #    yield np.array(image_batch)
        yield np.asarray(image_batch),to_categorical(np.asarray(labels),len(set(labellist)))

test_direc="/Users/siv.bala/Desktop/keras_data/datasets/cifar10/test/labels.txt"   
def test_data(directory):
    lines=[]
    with open(directory,'r') as f:
        lines=f.read().split("\n")
    imagelist=[]
    labellist=[]
    for item in lines[:-1]:
        imagelist.append((directory.replace("labels.txt","")+item.split(" ")[1]))
        labellist.append(int(item.split(" ")[0]))
    dic=dict(zip(imagelist, labellist))
    image_batch = []
    labels=[]
    for b in range(len(lines[:-1])):
        sample = imagelist[b]
        pic = Image.open(sample)
        image = np.asarray(Image.open(sample))
        
        image_batch.append(image.astype("float32")/255.0)
        labels.append(dic[sample])
    return np.asarray(image_batch),to_categorical(np.asarray(labels),len(set(labellist)))
    
dat,lab=test_data(test_direc)

#%%

#model loader from json file








