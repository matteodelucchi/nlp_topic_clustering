#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 10:54:10 2018

@author: matteo
"""
import os
import matplotlib.pyplot as plt
from keras.utils import plot_model

def model_visuals(model):
    '''
    Saves .png of model architecture in current working directory.
    Plots training & validation loss and accuracy (if labeled data is available).
    
    Arguments:
        model:   keras.engine.training.Model
    '''
    ### save graphical model visualisation in current working directory
    modelplot_path = str(os.getcwd()+"/"+model.name+".png")
    plot_model(model, to_file = modelplot_path)
          
    ### Accuracy only be evaluated in an unsupervised learning when labeled data is available
    ## evaluate the model
#    loss, accuracy = autoencoder.evaluate(x_train, x_train, verbose= int(VERBOSE_EVAL))
#    print('Accuracy: %f' % (accuracy*100))
    
    
#    # Plot training & validation accuracy values
#    plt.plot(model.history.history['acc'])
#    plt.plot(model.history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
    
    # Plot training & validation loss values
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()