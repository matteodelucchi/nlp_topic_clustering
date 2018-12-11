#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' 
Data Preprocessing
TODO:
    - error handling
    - README file
    - put weirdos, their replacements and stopwords in configfile
    - add all english named colors to stopwords
    - lemmatize german words (licence)
    
@author: matteodelucchi
'''
import pandas as pd

from bs4 import BeautifulSoup #BeautifulSoup - BeautifulSoup is a useful library for extracting data from HTML and XML documents
import re

import contractions # Contractions - Another simple library, solely for expanding contractions
import nltk # NLTK - The Natural Language ToolKit is one of the best-known and most-used NLP libraries in the Python ecosystem, useful for all sorts of tasks from tokenization, to stemming, to part of speech tagging, and beyond

import unicodedata
import inflect # Inflect - This is a simple library for accomplishing the natural language related tasks of generating plurals, singular nouns, ordinals, and indefinite articles, and (of most interest to us) converting numbers to words
from nltk.corpus import stopwords as stpwrds

from nltk.stem import LancasterStemmer, WordNetLemmatizer, snowball

####################################
### Create functions
####################################
def load_as_listoflist(FILENAME, partialize = False):
    '''loads file into a pandas dataframe and converts it to a python list of list'''
    text = pd.read_csv(FILENAME, 
                          sep='\t', 
                          usecols= [0,1,2])
    text = text.values.tolist()
    
    if partialize != False:
        # partialize file for debugging
        text = text[:partialize] 
    return text


def strip_html(text):
    '''Pulling data out of HTML structure'''
    for item in range(len(text)):
        for col in range(1,3): # long- and short description
            text[item][col] = BeautifulSoup(str(text[item][col]), "html.parser")
    return text

def remove_between_square_brackets(text):
    '''Data between [] is removed'''
    for item in range(len(text)):
        for col in range(1,3): # long- and short description
            text[item][col] = re.sub('\[[^]]*\]', '', str(text[item][col]))
    return text

def replace_umlaute(text):
    '''Replace ä,ö,ü,é,è,ß with ae, oe, ue, e, e, ss'''
    for item in range(len(text)):
        for col in range(1,3): # long- and short description
            uml = [u'ä', u'Ä', 
                   u"ö", u"Ö", 
                   u"ü", u"Ü", 
                   u"ß", 
                   u"é", u"É", 
                   u"è", u"È"]
            repl = ['ae', 'Ae', 
                    'oe', 'Oe', 
                    'ue', 'Ue', 
                    'ss', 
                    'e', 'E', 
                    'e', 'E']
            for umlaut in range(len(uml)):
                text[item][col] = re.sub(uml[umlaut], repl[umlaut], str(text[item][col]))
    return text
  
def denoise(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = replace_umlaute(text)
    return text


def replace_contractions(text):
    """Replace contractions (I'm to I am; -n't to not) in string of text"""
    for item in range(len(text)):
        for col in range(1,3): # long- and short description
            text[item][col] = contractions.fix(text[item][col])
    return text

def tokenize_text(text):
    '''Splits longer strings of text into smaller pieces, or tokens.
    Large chunks of text -> sentences; Sentences -> words.'''
    for item in range(len(text)):
            for col in range(1,3): # long- and short description
                text[item][col] = nltk.word_tokenize(text[item][col])
    return text


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    for item in range(len(words)): # each product
        for col in range(1,3): # long- and short description
            for word in range(len(words[item][col])): # each word
                words[item][col][word] = unicodedata.normalize('NFKD', words[item][col][word]).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    for item in range(len(words)): # each product
            for col in range(1,3): # long- and short description
                for word in range(len(words[item][col])): # each word
                    words[item][col][word] = words[item][col][word].lower()
    return words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    for item in range(len(words)): # each product
                for col in range(1,3): # long- and short description
                    for word in range(len(words[item][col])): # each word
                        '''replace punctuation with whitespace'''
                        words[item][col][word] = re.sub(r'[^\w\s]', '', words[item][col][word])
    return words  

def replace_numbers(words):
    """Replace all integer occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    for item in range(len(words)): # each product
        for col in range(1,3): # long- and short description
            for word in range(len(words[item][col])): # each word
                if words[item][col][word].isdigit(): 
                    words[item][col][word] = p.number_to_words(words[item][col][word])
    return words

def replace_weirdos(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    weirdos = ['2erknopfleiste']
    normalos = ['zweierknopfleiste']
    for item in range(len(words)): # each product
        for col in range(1,3): # long- and short description
            for word in range(len(words[item][col])): # each word
                for weird in range(len(weirdos)): # itearate over list of weirdos
                    if words[item][col][word] == weirdos[weird]:
                        '''if word is a weirdo, replace it with corresponding
                        normal word'''
                        words[item][col][word] = normalos[weird]
    return words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stopwords = stpwrds.words('english')
    stopwords_germ = stpwrds.words('german')
    stopwords.extend(stopwords_germ)
    newStopWords = ["gm", "fuer", "inklus", "ton", "gm²", "gqm", "gm2", "g/qm",
                    "beim", "inkl", "stueck", "inklus", "ueber", "zwei", "gross", 
                 "innen", "°c", "c", "moeglich", "cm", "x", "xcm",
                 "xs", "s", "m", "l", "xl", "xxl",
                 "grey","white", "black", "schwarz", "blaue", "weiss"] # TODO: add all english named colors
    stopwords.extend(newStopWords)
    for item in range(len(words)): # each product
        for col in range(1,3): # long- and short description
            for word in range(len(words[item][col])): # each word
                for stop in range(len(stopwords)): # itearate over list of stopwords
                    if words[item][col][word] == stopwords[stop]:
                        '''if word is a stopword, replace it with ""'''
                        words[item][col][word] = ''
    return words

def remove_whitespace(words):
    """Remove white space from list of tokenized words"""
    for item in range(len(words)): # each product
                for col in range(1,3): # long- and short description
                    for word in range(len(words[item][col])): # each word
                        while '' in words[item][col]: 
                            '''remove the whitespace'''
                            words[item][col].remove('')
    return words  

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = replace_weirdos(words)
    words = remove_stopwords(words)
    words = remove_whitespace(words)
    return words


def stem_en_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = words
    for item in range(len(stems)): # each product
        for col in range(1,3): # long- and short description
            for word in range(len(stems[item][col])): # each word
                stems[item][col][word] = stemmer.stem(stems[item][col][word])
    return stems

def stem_ge_words(words):
    """Stem words in list of tokenized words"""
    stemmer = snowball.GermanStemmer()
    stems = words
    for item in range(len(stems)): # each product
        for col in range(1,3): # long- and short description
            for word in range(len(stems[item][col])): # each word
                stems[item][col][word] = stemmer.stem(stems[item][col][word])
    return stems

def lemmatize_en_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = words
    for item in range(len(lemmas)): # each product
        for col in range(1,3): # long- and short description
            for word in range(len(lemmas[item][col])): # each word
                lemmas[item][col][word] = lemmatizer.lemmatize(lemmas[item][col][word], pos='v')
    return lemmas

def lemmatize_ge_verbs(words):
    ''' get licence for GermanNet: http://www.sfs.uni-tuebingen.de/GermaNet/index.shtml
        and use this python library: https://github.com/wroberts/pygermanet
        Alternatively use spaCy: https://spacy.io/models/de#de_core_news_sm
        according to this post: https://stackoverflow.com/questions/44522536/german-stemming-for-sentiment-analysis-in-python-nltk'''
    pass
#    """Lemmatize verbs in list of tokenized words"""
#    lemmatizer = WordNetLemmatizer()
#    lemmas = words
#    for item in range(len(stems)): # each product
#        for col in range(1,3): # long- and short description
#            for word in range(len(stems[item][col])): # each word
#                lemmas[item][col][word] = lemmatizer.lemmatize(lemmas[item][col][word], pos='v')
#    return lemmas


def stem_and_lemmatize(words):
    ENstems = stem_en_words(words)
    GEstems = stem_ge_words(words)
    ENlemmas = lemmatize_en_verbs(words)
    GElemmas = lemmatize_ge_verbs(words)
    return ENstems, GEstems, ENlemmas, GElemmas


    
####################################
### preprocessing procedure for
### messy online shop data
####################################
def cleanCSV(rawfilepath, outputfilepath, partialize = False, OUTPUT = False, stemandlem = False):
    ''' Put the cleaning procedure together specifically for the messy online shop data.
    Arguments:
        rawfilepath:    specified in the file config.ini
        outputfilepath: specified in the file config.ini
        partialize:     Takes any number between 0 and the number of rows from the raw data.
                        Particularly handy for debugging with a big messy file to 
                        save time. if specified as False, no partialization is done - 
                        hence the whole file is processed.
                        partialize = 10 takes the first ten rows.
        OUTPUT:         If True, a step wise output of the procedure is displayed. 
        stemandlem:     if True, german and english stemming and lemmatization of the
                        words are performed and returned as variables.'''
	# load raw_data
    text = load_as_listoflist(rawfilepath, partialize = False)
    if OUTPUT == True:
	    print(text)
	
	# denoise text
    text = denoise(text)
    if OUTPUT == True:
	    print(text)
	
	# tokenize text to get words
    text = replace_contractions(text)
    if OUTPUT == True:
	    print(text)
        
    words = tokenize_text(text)
    if OUTPUT == True:
	    print(words)

	# normalize words
    words = normalize(words)
    if OUTPUT == True:
	    print(words)

	# stem and lemmatize words
    if stemandlem == True:
        [ENstems, GEstems, ENlemmas, GElemmas] = stem_and_lemmatize(words)

	# put the words list-of-list construct into a pandas Dataframe
    df = pd.DataFrame.from_records(words, columns = ["Model", "Long_Description", "Short_Description"]) 
    if OUTPUT == True:
	    print(df)

	#Make each product-description pair to one string
    for row in range(len(df)):
	    df.iloc[:,1][row] = ' '.join(str(e) for e in df.iloc[:,1][row])
	    df.iloc[:,2][row] = ' '.join(str(e) for e in df.iloc[:,2][row])
    if OUTPUT == True:
	    print(df)
	
	# save as csv file
    df.to_csv(outputfilepath, sep='\t', header = True, encoding = 'utf8', index = False)
    if OUTPUT == True:
        print("Cleaned file saved to:", outputfilepath)
        
    # specify return values 
    if stemandlem == True:
        return ENstems, GEstems, ENlemmas, GElemmas
