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
import libembedding2weights as encoding2weights
import libKmeans as km
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
colname = 'Short_Description'

### one-hot-encode documents
VOCAB_SIZE_FACTOR = 5 #factor to reduce the probability of collisions from the hash function. Takes value >0
MAX_LENGTH_FACTOR= 5 #factor which is multiplied to no. of words of longest string in document. Takes value >0

### Embedding
# Model Architecture
OUTPUT_DIM = 8
# Model compiler
OPTIMIZER='adam'
LOSS='binary_crossentropy'
METRICS=['acc']
# Fit Model
EPOCHS=50
VERBOSE=0
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
VOCAB_SIZE = len(docs)*VOCAB_SIZE_FACTOR #much larger than needed to reduce the probability of collisions from the hash function.

padded_docs = onehotencode.one_hot_encode(docs, 
                                          vocab_size= VOCAB_SIZE, 
                                          max_length_factor= MAX_LENGTH_FACTOR)
####################################
### Embedding
####################################
# hyper-parameters
MAX_LENGTH = len(max(padded_docs, key=len)) #returns no. of words of longest string in list

# get weights of the embedding layer
embedding_weights = encoding2weights.compile_fit_weigths(data= padded_docs,
                                                         input_dim= VOCAB_SIZE,
                                                         output_dim= VOCAB_SIZE,
                                                         max_length= MAX_LENGTH,
                                                         optimizer=OPTIMIZER,
                                                         loss=LOSS,
                                                         metrics=METRICS,
                                                         epochs= EPOCHS,
                                                         verbose= VERBOSE)
# minimum biparitite maching
#comes here...

####################################
### clustering
####################################
km.kmeans(embedding_weights, n_clusters = k) 