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
    return sentiment_poi_2000


# this creates the topic model, topic number, and probability from list data
def topic_model(series):
    # create list for model
    emails_lemm = list(series)
    # Set UMPAP random_state 42 for reproducability
    umap_model = UMAP(n_neighbors=15, n_components=5,
                      min_dist=0.0, metric=‘cosine’, random_state=42)
    # create model object
    topic_model = BERTopic(umap_model = umap_model, language = ‘english’)
    # fit_transforms on lemmatized data
    topics, probs = topic_model.fit_transform(emails_lemm)
    return topics, probs, topic_model

# create df of topic docs
def topic_docs(topic_model):
    # change topics docs to list, then to dataframe text column as string
    docs_items = topic_model.get_representative_docs().items()
    docs_list = list(docs_items)
    docs_df = pd.DataFrame(docs_list)
    docs_df.rename(columns={0:‘topic’, 1:‘text’}, inplace=True)
    docs_df.text = docs_df.text.astype(‘str’)
    return docs_df (edited) 