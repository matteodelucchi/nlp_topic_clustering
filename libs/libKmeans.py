#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 14:40:40 2018

@author: matteo
"""
import numpy as np
from sklearn.cluster import KMeans

def kmeans(embedding_weights, n_clusters = 2):
    emb_w = np.transpose(embedding_weights) # transpose embedding weights to have it in the correct format for kmeans
    kmeans = KMeans(n_clusters= n_clusters) # k just guessed from visual data set inspection
    kmeans.fit(emb_w)
    return kmeans.predict(emb_w)