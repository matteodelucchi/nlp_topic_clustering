---
output: 
  html_document: 
    toc: yes
---


Cluster the Topics from Product Description
===================================
Clustering of product categories from products of an online shop based on the semantically sparse product description using DL techniques. 

The goal is to represent the categories of the products of an online shop through an unsupervised approach, through which tedious labelling of each product by hand can be replaced by the clustering of an embedding layer from a neural network.  
In this first experimental approach, we try extract the weights of an word embedding from an encoder. The distance in the embedding space for each combination of word vectors is calculated by a bipartite graph and then clustered with K-means. 
An other approach could be to apply a deep  embedding clustering analogous to: Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

## Getting Started
Follow the instructions from here till the end.

### Prerequisites
Make sure to work with Python3 or higher.
```
~$ python3 --version
Python 3.6.6
```

### Manual file-check
Make sure, that the columns in the raw csv-file containing the products and their descriptions are separated by tabs and not commas (```text.csv```).

### Configuration file
Change the Paths and filenames according to your system environment in
```config.ini```

### Main script file
In ```main.py```:  
  - Change your workingdirectory to the directory where ```config.ini``` is located.  
  - Set the variables in the section "Parameters" according to your needs and feelings :-)  
  - Run the section "Preprocess raw data" only if not already done or if you'd like to change something. Otherwise it can take quite a bit of time...  
  
### Run the programm
The goal is to only run ```main.py``` to get as output:  
- a cleaned .csv file
- stats about the NN
- clustering

## Data Preprocessing
By calling ```cleanCSV``` a messy .csv-file is processed to a cleaned .csv-file accordingly to be used in the later process.
```cleanCSV(rawfilepath, outputfilepath, partialize = False, OUTPUT = False, stemandlem = False)```  
**Arguments:**  
        rawfilepath:    specified in the file config.ini  
        outputfilepath: specified in the file config.ini  
        partialize:     Takes any number between 0 and the number of rows from the raw data. Particularly handy for debugging with a big messy file to save time. if specified as False, no partialization is done - hence the whole file is processed. partialize = 10 takes the first ten rows.  
        OUTPUT:         If True, a step wise output of the procedure is displayed.   
        stemandlem:     if True, german and english stemming and lemmatization of the words are performed and returned as variables.  

After loading the .csv-file with the raw data it proceeds as following:  
1. denoises text  
    - Pulling data out of HTML structure  
    - Data between [] is removed  
    - Replace Umlaute  
2. tokenization  
    - replace contractions
    - Splits longer strings of text into smaller pieces, or tokens. Large chunks of text -> sentences; Sentences -> words.  
3. normalizes words  
    - Remove non-ASCII characters  
    - Convert all characters to lowercase from list of tokenized words  
    - Remove punctuation  
    - Replace all integer occurrences with textual representation  
    - Replace weird words with other textual representation. i.e. "2erknopfleiste" to "zweierknopfleiste"  
    - Remove stop words. Predefined german and english stopwords and self-specified ones.  
    - Remove white space  
4. stemming & lemmatization  
    - stemm english and german words.
    - lemmatize english words. NOTE: not yet implemented for german words.  
5. saves the file as .csv and if specified, returns the german and english stems and lemmas.  

## One-Hot-Encode Documents
First of all, the text in the documents is converted to a sequence of words (or tokens). This is required, even though the same procedure was already done in *Data Preprocessing* since the documents are loaded again from a .csv file with the cleaned data.  
Then with ```keras.preprocessing.text.one_hot()``` which is a wrapper for the ```keras.preprocessing.text.hashing_trick()``` function, returns an integer encoded version of the document. The use of a hash function means that there may be collisions and not all words will be assigned unique integer values. Therefore the argument ```vocab_size``` is choosen much bigger than the actual length of the document (specified by the parameter ```VOCAB_SIZE_FACTOR```).  
Finally the sequences are padded to have all the same length. Which is basically adding zeros to the rows which have a size smaller than the longest row; considering again collision in the hash space by specifying  (specified by the parameter ```MAX_LENGTH_FACTOR```).




