# Getting Ahead of the Headline 
Analyzing employee emails using Natural Language Processing, Sentiment Analysis, and Time Series Analysis<br>

![](00_Stephanie/images/project_teaser.png)
Enron Topic Modeling | Codeup, Hopper Cohort | March 2022<br><br>
>Paige Guajardo<br>
Rajaram Gautam<br>
Stephanie Jones<br>
Kaniela Denis<br><br>


# About the Project
<p>In this unsupervised machine learning project we are exploring and analyzing Enron employee emails. We are using topic modeling, sentiment analysis, and time series analysis to identify trends in communication themes over time. For our MVP we are looking at a corpus of 5,575 emails sent by people of interest, as identified by the official congressional report on the role of Enron's board of directors (https://bit.ly/3Hjz5rI) on the collapse of the company. 

## Initial Hypothesis
Our initial hypothesis is that there will be distinct trends in email topics and sentiment over time. 

## Background
Enron Corporation was a major American energy, commodities, and services company that declared bankruptcy in December 2001 after over a decade of fraudulent accounting practices. During an error of more lenient financial regulations and high market speculation, Enron hid its financial losses in special purposes entities, making it appear much more profitable on paper than it actually was.
<br><br>
Enron has become synonymous with willful corporate fraud and corruption. The scandal also brought into question the accounting practices and activities of many corporations in the United States and was a factor in the enactment of the Sarbanes-Oxley Act of 2002. The scandal also affected the greater business world by causing the dissolution of the Arthur Andersen accounting firm, which had been Enron's main auditor for years.

## Business Goal
Company leaders, lawmakers, and the public will be able to use our analysis to identify key themes in communication between persons of interest in the early stages of investigating suspicious organizational activity. 

## Data Dictionary
variable | dtype | description
:-- | :-- | :--
`date` | datetime | date email was sent
`file` | object | email file path (storage)
`sender` | object | email sender email address
`subject` | object | text of email subject
`content` | object | raw email content
`lemmatize` | object | cleaned and lemmatized email content
`intensity` | float | vader sentiment intensity analyzer score
`polarity` | float | measure of email sentiment, -1 (neg) to 1 (pos)
`subjectivity` | float | measure of email subjectivity, 0 (obj), 1 (sub)
`poi` | bool | True == Person of Interest, someone connected to investigation (more on this below)
`is_internal` | bool | True == email was sent from Enron address

## Person of Interest
Using [this article]() from _________ we identified xX people as persons of interest because __________. 
Name | Connection to Enron | Enron Investigation
:-- | :-- | :---
Name | Role at Enron | Investigated/indicted/fired


# Data Science Pipeline 
## Planning
We used a [Trello Board](https://trello.com/b/osnQZqjJ/enronnlp-project) for planning.

## Data Wrangling
Data source: [Kaggle](https://www.kaggle.com/wcukierski/enron-email-dataset), Will Cukierski | 2016 

## Exploratory Analysis
1. How does employee sentiment change over time?
2. What percentage of emails were sent by employees connected to the investigation and are their average sentiment scores different from the overall average sentiment scores?
3. Were any emails sent from external addresses and, if so, how do their sentiment scores compare to interal emails' scores?
4. What senders had the highest and lowest sentiment scores?
5. What were common themes among emails sent by those who identified as  person of interest?
6. Were there any unique themes by year?

## Modeling
We used the `BERTopic` algorithm for topic modeling and looked at common topics for all emails from persons of interest by year, from 2000 to 2002. Initially, we planned to include 1999 but there were not enough emails for that year to generate topics from this algorithm. 

## Conclusions
If we had more time:
- Scale x3 sentiment scores
- Explore Nulls
- Explore Word Frequency Analysis with Email Subject
- Explore email recipients
- Explore and model with non-POI and POI emails (not just POI)
- Bin intensity, polarity, and subjectivity scores
- Explore clustering with the three different measures for sentiment
- Explore variables related to persons of interest, such as tenure with the company, salary, and investigation outcome

# Steps to Recreate
1. Read this README.md file<br>
2. Ensure you have latest version of Python installed<br>
3. Read email corpus to DataFrame (see link in citation below)<br>
4. Install and Import Python Libraries:

Library | Import Code | Documentation
:-- | :-- | :-- 
Pandas | `import pandas as pd` | https://pandas.pydata.org/docs/
Numpy | `import numpy as np` | https://numpy.org/doc/ 
email.parser | `from email.parser import Parser` | https://docs.python.org/3/library/email.parser.html
datetime | `import date_time` | https://docs.python.org/3/library/datetime.html
Vader, Sentiment Analysis | `from nltk.sentiment.vader import SentimentIntensityAnalyzer` | https://www.nltk.org/_modules/nltk/sentiment/vader.html 
Textblob, Sentiment Analysis | `from textblob import TextBlob` | https://pypi.org/project/textblob/
Bertopic, Topic Modeling | `from bertopic import BERTopic` | https://pypi.org/project/bertopic/

# References and Citations
Bert topic model:
>@misc{grootendorst2020bertopic,
  author       = {Maarten Grootendorst},
  title        = {BERTopic: Leveraging BERT and c-TF-IDF to create easily interpretable topics.},
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v0.9.4},
  doi          = {10.5281/zenodo.4381785},
  url          = {https://doi.org/10.5281/zenodo.4381785}
}

Wikipedia 
>- [Enron](https://en.wikipedia.org/wiki/Enron)
>- [Enron Scandal](https://en.wikipedia.org/wiki/Enron_scandal)
>- [Enron Corpus](https://en.wikipedia.org/wiki/Enron_Corpus)
>- [California Energy Crisis, 2000-2001](https://en.wikipedia.org/wiki/2000%E2%80%9301_California_electricity_crisis)

Data Source
>Will Cukierski, [Kaggle](https://www.kaggle.com/wcukierski/enron-email-dataset) (2015, May)

Congressional Report on the Role of Enron Board of Directors in Enron's Collapse 
>https://www.govinfo.gov/content/pkg/CPRT-107SPRT80393/html/CPRT-107SPRT80393.htm

