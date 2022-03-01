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
    senders = []
    subjects = []

    # loop through email messages
    for i in df.message:
        # parse and set message to email data type
        headers = Parser().parsestr(i)
        # get the body text of the email
        body = headers.get_payload()
        # get the date from email
        date = headers['Date']
        # get sender of email
        sender = headers['From']
        # get email subject
        subject = headers['Subject']
        # append date, body, sender, subjectg text to lists
        bodies.append(body)
        dates.append(date)
        senders.append(sender)
        subjects.append(subject)

    # Set lists to dataframes
    body_df = pd.DataFrame(bodies, columns = ['Content'])
    dates_df = pd.DataFrame(dates, columns = ['Date'])
    senders_df = pd.DataFrame(senders, columns = ['Sender'])
    subjects_df = pd.DataFrame(subjects, columns = ['Subject'])

    # Create a series of those colums
    content = body_df['Content']
    dates = dates_df['Date']
    send = senders_df['Sender']
    subj = subjects_df['Subject']

    # Insert those seriess into our orignal dataframe and create those columns
    df.insert(1, "content", content)
    df.insert(1, "date", dates)
    df.insert(1, "sender", send)
    df.insert(1, "subject", subj)

    # drop message 
    df = df.drop(columns = ['message'])

    return df

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Cleans emails
def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = string.lower()
    string = string.strip()
    string = string.replace('\n', ' ')
    string = string.replace('\t', ' ')
    string = string.replace('\r', ' ')
    string = unicodedata.normalize('NFKD', string)\
            .encode('ascii', 'ignore')\
            .decode('utf-8', 'ignore')
    string = re.sub(r"[^a-z0-9'\s]", "", string)
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
    
    #if os.path.isfile('emails.csv'):
    #    df = pd.read_csv('emails.csv')
    
    #else:
    df['clean'] = df[column].apply(basic_clean)\
                    .apply(tokenize)\
                    .apply(stem)

    df['tokenize'] = df[column].apply(basic_clean)\
                    .apply(tokenize)
                
    df['stop_words'] = df[column].apply(basic_clean)\
                    .apply(tokenize)\
                    .apply(remove_stopwords)

    '''
    df['stemm'] = df[column].apply(basic_clean)\
                    .apply(tokenize)\
                    .apply(remove_stopwords)\
                    .apply(stem)
    '''
    df['lemmatize'] = df[column].apply(basic_clean)\
                    .apply(tokenize)\
                    .apply(remove_stopwords)\
                    .apply(lemmatize)

    #df.to_csv('clean_emails.csv')
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
# ---------------------------------------------------------------
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# ---------------------------------------------------------------    

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
def create_poi_column(df):

    # creating poi list
    poi = ['andrew.fastow@enron.com',
    'richard.causey@enron.com',
    'rick.buy@enron.com',
    'ben.glisan@enron.com',
    'mary.joyce@enron.com',
    'jeff.skilling@enron.com',
    'jeffreyskilling@yahoo.com',
    'ronniechan@hanglung.com',
    'jhduncan@aol.com',
    'wgramm@aol.com',
    'wgramm@gmu.edu',
    'kenneth.lay@enron.com',
    'ken.lay-@enron.com',
    'ken.lay@enron.com',
    'ken.lay-.chairman.of.the.board@enron.com',
    'kevin_a_howard.enronxgate.enron@enron.net',
    'michael.krautz@enron.com',
    'rex.shelby@enron.com',
    'rex_shelby@enron.net',
    'james.brown@enron.com',
    'christopher.calger@enron.com',
    'tim.despain@enron.com',
    'kevin.hannon@enron.com',
    'mark.koenig@enron.com',
    'john.forney@enron.com',
    'ken.rice@enron.com',
    'ken_rice@enron.net',
    'paula.rieker@enron.com',
    'david.delainey@enron.com',
    'dave.delainey@enron.com',
    'jeff.richter@enron.com',
    'tim.belden@enron.com',
    'raymond.bowen@enron.com',
    'wes.colwell@enron.com',
    'dan.boyle@enron.com']

    df['poi'] = np.where(df.sender.isin(poi), True, False)
    return df
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def create_internal_column(df):
    #internal = df[df.sender.str.contains('@enron.com')]
    df['internal'] = np.where(df.sender.str.contains('@enron.com'), True, False)

    return df


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Creates base data frame with all these columns (date | content | clean | tokenize | stop_wards | lemmatize | intensity | subjectivity | polarity)
def create_topic_df():
    df = acquire_emails()

    # Clean Data base to (date | content | clean | tokenize | stop_wards | lemmatize)
    df = clean_emails(df)

    # Creates dataframe with added (intensity | subjectivity | polarity) columns
    df = create_scores(df)

    df = create_poi_column(df)

    df = create_internal_column(df)

    return df


# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Creates Time Series Data Frame after all the changes
def create_time_series_df(df):
    df = df.drop(columns = ['file', 'sender', 'subject', 'content', 'clean', 'tokenize', 'stop_words', 'lemmatize'])

    df.date = pd.to_datetime(df.date, utc=True)

    df = df.set_index('date').sort_index()

    return df

# ---------------------------------------------------------------
# ---------------------------------------------------------------
def time_series_df_final(df):
    df['year'] = df.index.year
    df['month'] = df.index.month

    df = df[df.year < 2003]
    df = df[df.year > 1998]
    
    return df    

# ---------------------------------------------------------------
# ---------------------------------------------------------------
# Uses all the functions and creates 2 dataframes for usage on topic modeling and time series analysis
def create_dataframes_wrangle():
    # Acquire Emails
    df = create_topic_df()

    # use the df and create the time series dataframe!
    time_series_df = create_time_series_df(df)

    # sets the time_series dataframe with columns (intensity | polarity | subjectivity | year | month) and sets to the years 1999 and 2002
    time_series_df = time_series_df_final(time_series_df)
    return df, time_series_df


# ---------------------------------------------------------------
# ---------------------------------------------------------------




