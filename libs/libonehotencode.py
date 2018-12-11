#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:15:39 2018

@author: matteo
"""

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence

def one_hot_encode(docs, vocab_size, max_length_factor):
    ''' First converts a text to a sequence of words (or tokens).
    Then with keras.preprocessing.text.one_hot() whic is a wrapper for the hashing_trick() 
    function, returns an integer encoded version of the document. 
    The use of a hash function means that there may be collisions and not all
    words will be assigned unique integer values.
    Finally pads sequences to the same length.'''
    wordsequence = [text_to_word_sequence(str(d)) for d in docs]
    encoded_docs = [one_hot(str(d), vocab_size) for d in wordsequence]
    padded_docs = pad_sequences(encoded_docs, 
                                maxlen = (len(max(encoded_docs, key=len))*max_length_factor), 
                                padding= 'post')
    return padded_docs