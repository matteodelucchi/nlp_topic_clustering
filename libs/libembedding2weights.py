#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:18:11 2018

@author: matteo
"""
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Dense


def embedding(input_dim= 50, output_dim = 8, max_length= 4):
    model = Sequential()
    model.add(Embedding(input_dim= input_dim, output_dim= output_dim, input_length= max_length))
    model.add(Flatten())
    model.add(Dense(max_length, activation='sigmoid')) #  output arrays of shape (*, MAX_LENGTH)
    return model

def compile_fit_weigths(data,
                        input_dim,
                        output_dim, 
                        max_length,
                        optimizer, 
                        loss, 
                        metrics,
                        epochs, 
                        verbose):
    '''Arguments:
        data= padded_docs,
        input_dim= VOCAB_SIZE,
        output_dim= VOCAB_SIZE, 
        max_length= MAX_LENGTH,
        optimizer=OPTIMIZER, 
        loss=LOSS, 
        metrics=METRICS,
        epochs= EPOCHS, 
        verbose= VERBOSE'''
    model = embedding(input_dim= input_dim, output_dim= output_dim, max_length= max_length)

    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # summarize the model
    print(model.summary())
    
    ## get weights of embedding layer for clustering
    np.shape(model.get_weights())

    ## fit the model
    history = model.fit(x= data, y= data, epochs= epochs, verbose= verbose) # The Network is supposed to just represent it's own data. Therefore, no labels are needed (unsupervised)
    
    ## evaluate the model
    #loss, accuracy = model.evaluate(sentences_onehot, labels, verbose= VERBOSE_EVAL)
    #print('Accuracy: %f' % (accuracy*100))
    
    ## predict classes of test_doc
    #predicted_classes = model.predict_classes(padded_test_docs)
    #print(predicted_classes, test_docs)
    return np.array(model.get_weights()[0]) # get weights of the embedding layer
