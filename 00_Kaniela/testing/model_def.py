#import wrangle_py
from bertopic import BERTopic
from copy import deepcopy

def bertopic_model_2000(df):

    df = df[df.year == 2000]

    df = df.drop(columns=['internal', 'year', 'month', 'polarity', 'subjectivity', 'intensity', 'date'])
    return