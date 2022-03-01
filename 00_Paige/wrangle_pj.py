import pandas as pd
import numpy as np

# USING IMPORT EMAILS TO PARSE THORUGH MESSAGES
import email
from email.parser import Parser

import unicodedata
import re
import json
import os

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
from time import strftime
#from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Intensity Score (sentiment score)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Polarity / Subjectivity scores
from textblob import TextBlob


# Acquire Emails (acquires dataset)
def acquire_emails():
    df = pd.read_csv('email.csv')

    bodies = []
    dates = []

    # loop through email messages
    for i in df.message:
        # parse and set message to email data type
        headers = Parser().parsestr(i)
        # get the body text of the email
        body = headers.get_payload()
        # get the date from email
        date = headers['Date']
        # append date and body text to lists
        bodies.append(body)
        dates.append(date)

    # Set lists to dataframes
    body_df = pd.DataFrame(bodies, columns = ['Content'])
    dates_df = pd.DataFrame(dates, columns = ['Content'])

    # Insert those data frames into our orignal dataframe
    df.insert(1, "content", body_df)
    df.insert(1, "date", dates_df)
    return df

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Cleans emails
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

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# tokenize clean
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

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# stemmanizes tokenize content
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

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# lemmatizes tokenize content
def lemmatize(string):
    """lemmatize [summary]
    Args:
        string ([type]): [description]
    Returns:
        [type]: [description]
    """

    wnl = WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(lemmas)
    return  string
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# remove stop words (elmimates words via stopword database import)
def remove_stopwords(string, extra_words=None, exclude_words=None):
        """remove_stopwords [summary]
        Args:
            string ([type]): [description]
            extra_words ([type], optional): [description]. Defaults to None.
            exclude_words ([type], optional): [description]. Defaults to None.
        Returns:
            [type]: [description]
        """
        stopw = stopwords.words('english')

        if extra_words:
            stopw.append(word for word in extra_words)

        elif exclude_words:
            stopw.remove(word for word in exclude_words)

        words = string.split()
        filtered_words = [word for word in words if word not in stopw]
        string = ' '.join(filtered_words)
        return string

# ---------------------------------------------------------------
# ---------------------------------------------------------------
###### MAIN CLEANING FUNCTION
# Takes in dataframe first created and uses all functions to create a clean text of the emails/tokenize of it/ and stem of it

def clean_emails(df, column = 'content'):
    '''
    Creates DF with cleaned version of emails/ stem version of clean emails / lemetized versions of clean emails
    '''
    
    if os.path.isfile('emails.csv'):
        df = pd.read_csv('emails.csv')
    
    else:
        df['clean'] = df[column].apply(basic_clean)\
                        .apply(tokenize)\
                        .apply(stem)

        df['tokenize'] = df[column].apply(basic_clean)\
                        .apply(tokenize)
                
        df['stop_words'] = df[column].apply(basic_clean)\
                        .apply(tokenize)\
                        .apply(remove_stopwords)

        df['stemm'] = df[column].apply(basic_clean)\
                        .apply(tokenize)\
                        .apply(remove_stopwords)\
                        .apply(stem)

        df['lemmatize'] = df[column].apply(basic_clean)\
                        .apply(tokenize)\
                        .apply(remove_stopwords)\
                        .apply(lemmatize)

        df.to_csv('clean_emails.csv')
    return df

# ---------------------------------------------------------------
# ---------------------------------------------------------------



# ---------------------------------------------------------------
# ---------------------------------------------------------------

# sentiment scoring function creates polarity and subjectivity
def sentiment_scores(string):
    '''
    This function takes in a string of text and applies the textblob function, returning 
    a score for polarity and subjectivity.
    
        - Polarity (float; -1, 1) negative, nuetral, positive sentiment
        - Subjectivity (float; 0, 1) 0, most objective // 1, most subjective
    '''
    
    polarity, subjectivity = TextBlob(str(string)).sentiment
    
    return polarity, subjectivity

# function to add textblob sentiment scores to df
def add_scores(df, clean_msg_col):
    '''
    This function takes in a df and column of strings to apply the sentiment_scores
    textblob function to. It returns a df with the polarity and subjectivity scores added.
    '''
    
    df['polarity, subjectivity'] = df[clean_msg_col].apply(sentiment_scores)
    
    pol = []
    subj = []
    for tuple_ in df['polarity, subjectivity']:
        pol.append(list(tuple_)[0])
        subj.append(list(tuple_)[1])
    
    print('polarity and subjectivity algo complete')
    # df = df.drop(columns = ['polarity, sentiment'])
    df['polarity'] = pol
    df['subjectivity'] = subj
          
    print('added sub and pol to df')
    
    # dropping polarity, subjectivity col
    df.drop(columns = ['polarity, subjectivity'], inplace = True)
    
    return df

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Creates intensity/ polarity / subjectivity columns for database
def create_scores(df):

    # uses sentiment analyzer to create intensity scores column
    sia = SentimentIntensityAnalyzer()
    df['intensity'] = df.lemmatize.apply(lambda doc: sia.polarity_scores(doc)['compound'])

    # uses add_score function to create polarity and subjectivity column
    df = add_scores(df, 'lemmatize')
    return df

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Creates Time Series Data Frame after all the changes
def create_time_series_df(df):
    df = df.drop(columns = ['file', 'message', 'sender', 'subject', 'content', 'clean', 'tokenize', 'stop_words', 'lemmatize', 'stemm'])

    df.date = pd.to_datetime(df.date, utc=True)

    df = df.set_index('date').sort_index()

    return df

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Uses all the functions and creates 2 dataframes for usage on topic modeling and time series analysis
def create_dataframes_wrangle():
    # Acquire Emails
    df = acquire_emails()

    # Clean Data base to (date | content | clean | tokenize | stop_wards | stemm | lemmatize)
    df = clean_emails(df)

    # Creates dataframe with added (intensity | subjectivity | polarity) columns
    df = create_scores(df)

    # use the df and create the time series dataframe!
    time_series_df = create_time_series_df(df)

    return df, time_series_df
