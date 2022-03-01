#import wrangle_py
from bertopic import BERTopic
from copy import deepcopy

#---------------------------------------------------
# Generate Model from data frame for emails in 2000
#---------------------------------------------------
def bertopic_model_2000(df):

    # Create Data Frame Only in 2000
    df = df[df.year == 2000]

    # Drop unneaded columns for model
    df = df.drop(columns=['internal', 'year', 'month', 'polarity', 'subjectivity', 'intensity', 'date'])

    df = df[df.poi == True]

    # Create list of emails from 2000
    emails = list(df.lemmatize)


    # Create umap for reproducibility
    umap_model = UMAP(n_neighbors=15, n_components=5, 
                  min_dist=0.0, metric='cosine', random_state=42)

    # Create topics by using BERTopic algorithm
    topic_model_2000 = BERTopic(umap_model = umap_model, language = 'english')

    topics , probs = topic_model_2000.fit_transform(emails)


    return topic_model_2000, topics, probs

#---------------------------------------------------
# Generate Model from data frame for emails in 2001
#---------------------------------------------------
def bertopic_model_2001(df):

    # Create Data Frame Only in 2000
    df = df[df.year == 2001]

    # Drop unneaded columns for model
    df = df.drop(columns=['internal', 'year', 'month', 'polarity', 'subjectivity', 'intensity', 'date'])

    df = df[df.poi == True]

    # Create list of emails from 2000
    emails = list(df.lemmatize)


    # Create umap for reproducibility
    umap_model = UMAP(n_neighbors=15, n_components=5, 
                  min_dist=0.0, metric='cosine', random_state=42)

    # Create topics by using BERTopic algorithm
    topic_model_2001 = BERTopic(umap_model = umap_model, language = 'english')

    topics , probs = topic_model_2001.fit_transform(emails)

    return topic_model_2001, topics, probs
    
#---------------------------------------------------
# output a hiearchy visual of a model
#---------------------------------------------------
def model_hierachy_chart(model):
    model.visualize_hierachy()

#---------------------------------------------------
#---------------------------------------------------
def model_dataframe(model):
    sentiment_poi_2000 = model.get_topic_info()