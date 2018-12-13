''' 
Topic Extraction from messy online shop data.

TODO:
    - error handling
    - README file
    
@author: matteodelucchi
'''
####################################
### Housekeeping
####################################
import sys
import os
import configparser
import pandas as pd
from sklearn.model_selection import train_test_split


# change working directory to path of project
os.chdir("/home/matteo/polybox/MSc_ACLS/topic_modeling_project") 

# Parse config file
config = configparser.ConfigParser()
config.read('config.ini')

# ad directory of the libraries to PYTHONPATH
sys.path.append(config['SYSTEMSETUP']['LIBS'])

# load own modules
import libpreprocess as preprocess
import libonehotencode as onehotencode
import libKmeans as km
import libautoencoder as ac
import libvisual as vis
####################################
### Parameters
####################################
### Preprocessing
do_preprocessing = False # Boolean. If True, the raw file will be cleaned. preprocessed and saved in a new clean csv file
raw_file = config['PREPROCESSING']['RAW_FILE_ALL'] # filepath to messy data file
clean_file = config['PREPROCESSING']['CLEAN_FILE'] # filepath to cleaned data file
PARTIALIZE = False # False or integer with the amount of observables to consider.

### Import cleaned data
# choose which column to take as document corpus
colname = 'Short_Description' #Arguments: 'Short_Description' or 'Long_Description'

### one-hot-encode documents
VOCAB_SIZE_FACTOR = 5 #factor to reduce the probability of collisions from the hash function. Takes value >0
MAX_LENGTH_FACTOR= 5 #factor which is multiplied to no. of words of longest string in document. Takes value >0

### Embedding
# Model Architecture
OUTPUT_DIM = 8
# Model compiler
OPTIMIZER='adam' #'adadelta', 'adam', SGD(lr=0.1, decay=0, momentum=0.9) # from keras.optimizers import SGD
LOSS='binary_crossentropy' # 'mse', 'kullback_leibler_divergence', 'categorical_crossentropy'
# Fit Model
EPOCHS=50
BATCH_SIZE=200
# Model Evaluation
VERBOSE_EVAL=0

### clustering
k = 17  # k is guessed from visual data set inspection
####################################
### Preprocess raw data
####################################
if do_preprocessing:
    preprocess.cleanCSV(rawfilepath = raw_file, 
                    outputfilepath = clean_file, 
                    partialize = PARTIALIZE, 
                    OUTPUT = False, 
                    stemandlem = False)

####################################
### import cleaned data file 
####################################
# import documents
df=pd.read_csv(clean_file, sep='\t', header = 0)

# select the specified column to take as document corpus
docs= df[colname]
print(docs[:10])

####################################
### one-hot-encode documents
####################################
VOCAB_SIZE = int(len(docs)*VOCAB_SIZE_FACTOR) #much larger than needed to reduce the probability of collisions from the hash function.

padded_docs = onehotencode.one_hot_encode(docs, 
                                          vocab_size= VOCAB_SIZE, 
                                          max_length_factor= MAX_LENGTH_FACTOR)

# split data in train and test:
x_train, x_test = train_test_split(padded_docs, 
                                   test_size = 0.33, 
                                   random_state=42)

####################################
### Embedding
####################################
### hyper-parameters
input_dim = int(padded_docs.shape[1]) #returns no. of words of longest string in list
# this is the size of our encoded representations
encoding_dim = int(padded_docs.shape[1] / 50) # reduction of factor 50
# size of the output dimension (normally in autoencoder = input_dim)
output_dim = input_dim




### simple autoencoder
autoencoder = ac.onelayer_autoencoder(input_dim = input_dim, 
                                      encoding_dim = encoding_dim, 
                                      output_dim = output_dim)
autoencoder.model.compile(loss='categorical_crossentropy', optimizer='adam')
autoencoder.model.fit(x_train, x_train,
                      epochs=EPOCHS,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      validation_data=(x_test, x_test))
# encode and decode some words
# note that they are taken from the *test* set
encoded_words = autoencoder.encoder.predict(x_test)
decoded_words = autoencoder.decoder.predict(encoded_words)
# plot model loss and save picture of model architecture
vis.model_visuals(autoencoder.model)




### Deep autoencoder...
deepautoencoder = ac.deep_autoencoder(input_dim = input_dim,
                                  output_dim = output_dim)
deepautoencoder.model.compile(loss=LOSS, optimizer=OPTIMIZER)
deepautoencoder.model.fit(x_train, x_train,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          validation_data=(x_test, x_test))
# encode and decode some words
# note that they are taken from the *test* set
encoded_words = deepautoencoder.encoder.predict(x_test)
decoded_words = deepautoencoder.decoder.predict(encoded_words)
# plot model loss and save picture of model architecture
vis.model_visuals(deepautoencoder.model)
### get the weights of the most dense encoded layer:
#taking a look at the model:
print(deepautoencoder.model.summary())
weights = deepautoencoder.model.layers[4].get_weights()[0] #layer.get_weights()[0] to access the weights and 1 for the biases

### TO BE CONTINUED FROM HERE ON....


####################################
### minimum biparitite maching
####################################
#comes here...

####################################
### clustering
####################################
#km.kmeans(embedding_weights, n_clusters = k) 