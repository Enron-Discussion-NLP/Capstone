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

import wrangle

# bert topic modeling library
from bertopic import BERTopic

# import umap for reproducability
from umap import UMAP

def create_topic_model(df, column = 'lemmatize'):
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
    emails_lemm = list(df[column])
    
    # Set UMPAP random_state 42 for reproducability
    umap_model = UMAP(n_neighbors=15, n_components=5,
                      min_dist=0.0, metric='cosine', random_state=42)

    #creating the BERTopic model object
    topic_model = BERTopic(umap_model = umap_model, language = 'english')

    #fitting model object to lemmatized text and transforming to create topics
    topics, probs = topic_model.fit_transform(emails_lemm)

    # create df of topics with topic, count, and name
    topics_df = topic_model.get_topic_info()[1:]
    topics_df = topics_df.rename(columns={'Topic':'topic', 'Count':'count', 'Name':'name'})
    
    return topics, probs, topic_model, topics_df, emails_lemm



def create_topic_docs(topic_model):
    '''
    This function takes the topic_model and extracts the emails that were
    grouped to each topic. Then it takes each grouped docs that is in dictionary 
    data type and makes into a list. That list is made into a dataframe, columns 
    are renamed for readability, and each row in the text column is 
    made into a string
    '''
    # change topics docs to list, then to dataframe text column as string
    docs_items = topic_model.get_representative_docs().items()
    
    # sets the docs of topics into a list
    docs_list = list(docs_items)
    
    # sets list into data frame
    docs_df = pd.DataFrame(docs_list)
    
    # renames dateframe columns for readability
    docs_df.rename(columns={0:'topic', 1:'lemmatize'}, inplace=True)
    
    # text column is set as string
    docs_df.lemmatize = docs_df.lemmatize.astype('str')
    
    return docs_df



def create_topic_scores(df):
    '''
    This function runs create_topic_model function to get the topic_model and other objects.
    Topic model is used to extract the email text from each topic. It runs create_scores function
    from wrangle to get intensity, subjectivity, and polarity scores for each topic based on docs_df.
    Then is merges docs_df with scores and the topics_df that has topic name.
    '''
    
    # runs create_topic_model function to return topic_model objects
    topics, probs, topic_model, topics_df, emails_lemm = create_topic_model(df, column = 'lemmatize')
    
    # runs create_topic_docs to get docs_df which is the df with all docs as string
    docs_df = create_topic_docs(topic_model)
    
    # runs create scores from wrangle file to add intensity, subjectivity, and polarity
    docs_df_scores = wrangle.create_scores(df=docs_df)
    
    # merges the df with topic scores with the df of topics 
    topics_scores = topics_df.merge(docs_df_scores, on='topic', how='left')
    
    return topics, probs, topic_model, topics_df, docs_df, topics_scores, emails_lemm



def create_topic_scores_reduced(emails_lemm, topics, topic_model, i):
    '''
    This function uses a the BERTopic model with a reduced number of topics and creates
    docs_df
    '''
    
    # reduce number of topics to count depending previous heirarical grouping
    topic_model.reduce_topics(emails_lemm, topics, nr_topics=i)
    
    
    # create df of topics with topic, count, and name
    topics_df = topic_model.get_topic_info()[1:]
    topics_df = topics_df.rename(columns={'Topic':'topic', 'Count':'count', 'Name':'name'})
    
    # runs create_topic_docs to get docs_df which is the df with all docs as string
    docs_df = create_topic_docs(topic_model)
    
    # runs create scores from wrangle file to add intensity, subjectivity, and polarity
    docs_df_scores = wrangle.create_scores(df=docs_df)
    
    # merges the df with topic scores with the df of topics 
    topics_scores = topics_df.merge(docs_df_scores, on='topic', how='left')
    
    return topic_model, topics_df, docs_df, topics_scores



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
    topic_model = topic_model(df)

    #sets figure size
    plt.figure(figsize = figsize_)

    sns.barplot(data = pd.DataFrame(topic_model.get_topic(topic = topic_num)), y = 0, x = 1, palette = palette_)
    plt.title(title)
    plt.show()

def plot_distance_map(topic_model):
    '''
    This function takes in a fit and transformed model object and plots a
    topic distance map with each topic in one of four quadrants.

    This will allow us to see overlapping topics, outliers, and topic groups.
    '''
    #calling method to plot intertopic distance map
    return topic_model.visualize_topics()

def topic_tree(topic_model):
    '''
    This function takes in a fit and transformed model object and plots a 
    tree, with the levels of heirachical clustering.

    This will allow us to look at groups of similar topics at various
    levels of subgrouping.
    '''

    return topic_model.visualize_hierarchy()
