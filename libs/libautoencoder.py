#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:30:57 2018

@author: matteo
"""
import numpy as np
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Dense, RepeatVector, Input, TimeDistributed

    
def autoencoder(vocab_size = VOCAB_SIZE, src_txt_length = VOCAB_SIZE, sum_txt_length = (VOCAB_SIZE/(VOCAB_SIZE*4))):
    # encoder input model
    inputs = Input(shape=(src_txt_length,))
    encoder1 = Embedding(vocab_size, 128)(inputs)
    encoder2 = LSTM(128)(encoder1)
    encoder3 = RepeatVector(sum_txt_length)(encoder2)
    # decoder output model
    decoder1 = LSTM(128, return_sequences=True)(encoder3)
    outputs = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder1)
    # tie it together
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def fit_and_weights(model, data, epochs, verbose):
      ## fit the model
    model.fit(x= data, y= data, epochs= epochs, verbose= verbose) # The Network is supposed to just represent it's own data. Therefore, no labels are needed (unsupervised)
    
    return np.array(model.get_weights()) # get weights of the embedding layer
