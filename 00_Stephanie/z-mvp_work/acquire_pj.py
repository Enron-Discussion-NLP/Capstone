import pandas as pd
import numpy as np

# USING IMPORT EMAILS TO PARSE THORUGH MESSAGES
import email
from email.parser import Parser

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

