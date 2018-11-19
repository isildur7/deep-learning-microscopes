# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:20:44 2018

@author: Amey Chaware 

Input Data Pipeline for Overlapped Malaria Classification
Uses tf.data module
"""
#%%
import tensorflow as tf
import scipy.io as spio
import numpy as np

#%%
def to_one_hot(y,numcategories = 2):
    # converts a vector with numcategories categories to a one-hot matrix
    y_ = np.zeros((y.size,numcategories))
    y_[np.arange(y.size),y] = 1
    return y_

#%%
def load_data_malaria(directory, params):
    # directory containing the data stacks for malaria and no malaria
    # split: what fraction is training?
    # load data (0: uninfected, 1: infected):
    split = params.split
    X0 = spio.loadmat(directory+'images_noinfection_thicksmear_colorcombo.mat')
    X0 = X0['images_noinfection_thicksmear_colorcombo'].astype(np.float32)
    X1 = spio.loadmat(directory+'images_infected_thicksmear_colorcombo.mat')
    X1 = X1['images_infected_thicksmear_colorcombo'].astype(np.float32)
    
    # take only the center image:
    X0 = X0[:,:,[0,41,82],:]
    X1 = X1[:,:,[0,41,82],:]
    
    # reshape to num examples by row by column by channels
    X0 = np.moveaxis(X0, -1, 0)
    X1 = np.moveaxis(X1, -1, 0)
    y0 = np.zeros(X0.shape[0],dtype=np.int32)
    y1 = np.ones(X1.shape[0],dtype=np.int32)

    # split into training and validation, but don't mix the categories together
    # (they will be mixed during the feed_dict generation)
    X0train = X0[:int(split*len(X0))]
    X0val = X0[int(split*len(X0)):]

    y0train = y0[:int(split*len(X0))]
    y0val = y0[int(split*len(X0)):]

    X1train = X1[:int(split*len(X1))]
    X1val = X1[int(split*len(X1)):]

    y1train = y1[:int(split*len(X1))]
    y1val = y1[int(split*len(X1)):]

    return X0train,y0train,X1train,y1train,X0val,y0val,X1val,y1val

#%%

def get_iter_from_raw(data, params, numsuper=9,intensity_scale=1/9, training=True,):
    # data is the output of load_data_malaria
    # numsuper is the number of images summed, where one is malaria-infected
    # intensity_scale: tune this value so that the detector doesn't saturate
    # training specifies whether to generate from the training data or validation data
    # params is a dictionary storing the hyperparameters
    # returns a tf.iterator for training (or validation)
    
    X0train,y0train,X1train,y1train,X0val,y0val,X1val,y1val = data
    
    if training:
        X0 = X0train
        y0 = y0train
        X1 = X1train
        y1 = y1train
        number = int(params.number*params.split)
    else:
        X0 = X0val
        y0 = y0val
        X1 = X1val
        y1 = y1val
        number = int((1-params.split)*params.number)
    
    s0 = X0.shape
    s1 = X1.shape
    
    # make half of the batch no malaria, half with malaria
    inds0 = np.random.choice(s0[0],size=number*numsuper-number//2) #for the non-malaria
    inds1 = np.random.choice(s1[0],size=number//2) #for the malaria; only 1 malaria per stack; for half of the batch
    
    # examples for no malaria in stack
    X0stack = X0[inds0[:number//2*numsuper]].reshape(number//2,numsuper,s0[1],s0[2],s0[3])
    X0stack = X0stack.sum(1) #the superpositioned image
    
    # examples for one malaria in stack
    X1stack0 = X0[inds0[number//2*numsuper:]].reshape(number//2,numsuper-1,s0[1],s0[2],s0[3]) #from the nonmalaria set
    X1stack1 = X1[inds1]
    X1stack = X1stack0.sum(1) + X1stack1
    
    # stack them all together
    Xbatch = np.concatenate([X0stack,X1stack],axis=0)*intensity_scale
    Xbatch = np.minimum(Xbatch,255).astype(np.uint8).astype(np.float32) # threshold and discretize to 8-bit
    ybatch = np.concatenate([np.zeros(number//2,dtype=np.int32),np.ones(number//2,dtype=np.int32)])
    ybatch = to_one_hot(ybatch)
    
    # create dataset to be fed in thr pipeline
    dataset = tf.data.Dataset.from_tensor_slices((Xbatch, ybatch))
    dataset = dataset.shuffle(buffer_size=len(ybatch),reshuffle_each_iteration=True) # random shuffle in every iteration 
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1) # always keep one batch ready
    
    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
