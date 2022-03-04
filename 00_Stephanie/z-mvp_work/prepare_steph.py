import unicodedata
import re
import json
import os

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
import acquire_pj
from time import strftime

from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

###### Functions that will feed into Main cleaner function ######

def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = unicodedata.normalize('NFKD', string)\
            .encode('ascii', 'ignore')\
            .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()

    return string

def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    # Create tokenizer.
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    # Use tokenizer
    string = tokenizer.tokenize(string, return_str = True)
    
    return string


def stem(string):
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    
    # Use the stemmer to stem each word in the list of words we created by using split.
    stems = [ps.stem(word) for word in string.split()]
    
    # Join our lists of words into a string again and assign to a variable.
    string = ' '.join(stems)
    
    return string

###### MAIN CLEANING FUNCTION
def clean_emails(df, column = 'content'):
    '''
    [add docstring]
    '''
    
    if os.path.isfile('emails.csv'):
        df = pd.read_csv('emails.csv')
    
    else:
        df['clean'] = df[column].apply(basic_clean)\
                        .apply(tokenize)\
                        .apply(stem)
        df.to_csv('emails.csv')

    return df

#######
def split_data(df):
    '''
    
    '''
    train, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train, test_size = .3, random_state = 123)
    
    return train, validate, test
