#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 16:30:57 2018
Sources: 
    https://blog.keras.io/building-autoencoders-in-keras.html
    https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/
    https://stackoverflow.com/questions/47735205/keras-autoencoder-for-text-analysis

@author: matteo
"""
import numpy as np
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed

class ReturnValue(object):
    '''
    Creates a class object to easily access the return values from the models by
    
    For example to access the model itself:
        libautoencoder.onelayer_autoencoder(*kargs).model
    '''
    def __init__(self, mod, enc, dec):
        self.model = mod
        self.encoder = enc
        self.decoder = dec
     
def onelayer_autoencoder(input_dim, encoding_dim, output_dim):
    '''
    Creates a simple one layer keras autoencoder model. 

    
    Arguments:
        input_dim:      shape of the word padding which is basically the number
                        of words from the largest document (here: product description)
        encoding_dim:   shape of the encoding layer. input_docs reduced by a certain
                        reduction factor
        output_dim:     in an autencoder, the size of the input vector (input_docs).
    
    
    References: 
        https://blog.keras.io/building-autoencoders-in-keras.html
    '''
    # this is our input placeholder
    input_docs = Input(shape=(input_dim,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_docs)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(output_dim, activation='sigmoid')(encoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input_docs, decoded)
    
    # this model maps an input to its encoded representation
    encoder = Model(input_docs, encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    
    return ReturnValue(mod= autoencoder, enc= encoder, dec= decoder)

def deep_autoencoder(input_dim, output_dim):
    '''
    Creates a deep, 6 layer keras autoencoder model. 

    
    Arguments:
        input_dim:      shape of the word padding which is basically the number
                        of words from the largest document (here: product description)
        output_dim:     in an autencoder, the size of the input vector (input_docs).
    
    
    References: 
        https://blog.keras.io/building-autoencoders-in-keras.html
    '''
    input_docs = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_docs)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(output_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_docs, decoded)
    return ReturnValue(mod=autoencoder, enc= autoencoder, dec= autoencoder)






    '''
    LSTM need 3 dimensions. I still need some time to think about this.
    Below from here are just some non-successful trials and notes...
    '''
#### let's try seq2seq autoencoder
##same source: https://blog.keras.io/building-autoencoders-in-keras.html
#from keras.layers import Input, LSTM, RepeatVector
#from keras.models import Model
#
#timesteps = int(padded_docs.shape[0])
#input_dim = int(padded_docs.shape[1])
#latent_dim = int(padded_docs.shape[1] / 50) # reduction of factor 50
#
#inputs = Input(shape=(timesteps, int(padded_docs.shape[0]), int(padded_docs.shape[1])))
#encoded = LSTM(latent_dim)(inputs)
#
#decoded = RepeatVector(timesteps)(encoded)
#decoded = LSTM(input_dim, return_sequences=True)(decoded)
#
#sequence_autoencoder = Model(inputs, decoded)
#encoder = Model(inputs, encoded)
#
#encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#encoder.fit(x_train, x_train,
#                epochs=13,
#                batch_size=256,
#                shuffle=True,
#                validation_data=(x_test, x_test))
#
## encode and decode some words
## note that we take them from the *test* set
#encoded_words = encoder.predict(x_test)
#decoded_words = decoder.predict(encoded_words)




### try this: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html




def fit_and_weights(model, data, epochs, verbose):
      ## fit the model
    model.fit(x= data, y= data, epochs= epochs, verbose= verbose) # The Network is supposed to just represent it's own data. Therefore, no labels are needed (unsupervised)
    
    return np.array(model.get_weights()) # get weights of the embedding layer


