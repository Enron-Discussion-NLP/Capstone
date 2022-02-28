# python data manipulation and analysis library
import numpy as np

# python library supporting operations on multidimensional arrays and matrices 
import pandas as pd

# python plotting library for visualizations
import matplotlib.pyplot as plt

# data visualization library for python based on matplotlib providing higher level visual interface and options
import seaborn as sns

# formatting for notebook grids
plt.style.use('seaborn-whitegrid')

from sklearn.feature_extraction.text import CountVectorizer

# bert topic modeling library
from bertopic import BERTopic

def topic_model(df):
    '''
    This function takes in a df and fits the BERTopic model object on the lemmatized text,
    tranforming the raw text into topic groups, returning the transformed model object that 
    can be applied to various exploratory and model visualization methods. 
    
    This will allow us to call the model object on the topic modeling and visualization functions for a given
    df's list of text column strings.

    Args:
    - df

    Returns:
    - model: fit and transformed model object
    '''

    #creating a list of strings for each row of lemmatized text in df
    emails = list(df.lemmatize)

    #creating the BERTopic model object
    model = BERTopic(language = 'english', nr_topics = 'auto')

    #fitting model object to lemmatized text and transforming to create topics
    topics, probs = model.fit_transform(emails)

    return model

def get_topics(df):
    '''
    This function takes in a fit and transformed BERTopic model object and returns
    a df with the modeled topics. 

    This will allow us to look at all of the modeled topics for a given df's list of 
    column strings of text.
    '''
    #calling the topic_model() function to get fit and transformed model object
    model = topic_model(df)

    #df of model topics
    topics = model.get_topic_info()

    return topics


def plot_topic(model, topic_num, figsize_ = None, palette_ = None, title = None):
    '''
    This function plots a barchart for a single given topic's top words with optional
    chart formatting objects.

    This will allow us to visualize the top words for a given topic number and their
    frequencies.

    Args:
    - model: fit and transformed model object
    - topic_num: topic number (see model.get_topic_info() for df of topics with numbers)
    - figsize --> (width, height): [optional] set figure width and height
    - palette --> [list of x10 color codes: strings] [optional] bar colors
    - title --> (str) [optional] bar chart title
    '''

    #calling the topic_model() function to get fit and transformed model object
    model = topic_model(df)

    #sets figure size
    plt.figure(figsize = figsize_)

    sns.barplot(data = pd.DataFrame(model.get_topic(topic = topic_num)), y = 0, x = 1, palette = palette_)
    plt.title(title)
    plt.show()

def plot_distance_map(model):
    '''
    This function takes in a fit and transformed model object and plots a
    topic distance map with each topic in one of four quadrants.

    This will allow us to see overlapping topics, outliers, and topic groups.
    '''
    #calling method to plot intertopic distance map
    model.visualize_topics()

def topic_tree(model):
    '''
    This function takes in a fit and transformed model object and plots a 
    tree, with the levels of heirachical clustering.

    This will allow us to look at groups of similar topics at various
    levels of subgrouping.
    '''

    model.visualize_hierarchy()
