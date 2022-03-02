import numpy as np
from matplotlib import pyplot as plt


# A function that creates a time_series line graph for years 1999-2002 and respective values for sentiment
def time_series_graph_sentiment(df):
    train = df[df.year == 1999]
    highlight1 = df[df.year == 2000]
    highlight2 = df[df.year == 2001]
    test = df[df.year == 2002]

    # show  highlighted years with sentiment
    train_by_date = train.groupby(['date']).sentiment.sum().reset_index()

    highlight1_by_date = highlight1.groupby(['date']).sentiment.sum().reset_index()

    highlight2_by_date = highlight2.groupby(['date']).sentiment.sum().reset_index()

    test_by_date = test.groupby(['date']).sentiment.sum().reset_index()

    # Graph dates
    time_graph = plt.figure(figsize = (16, 10))
    plt.title("Sentiment Line Graph")
    plt.plot(train_by_date.date, train_by_date.sentiment)
    plt.plot(highlight1_by_date.date, highlight1_by_date.sentiment)
    plt.plot(highlight2_by_date.date, highlight2_by_date.sentiment)
    plt.plot(test_by_date.date, test_by_date.sentiment)
    plt.show()

# line graph for subjectivity time series
def time_series_graph_subjectivity(df):
    train = df[df.year == 1999]
    highlight1 = df[df.year == 2000]
    highlight2 = df[df.year == 2001]
    test = df[df.year == 2002]

    # show  highlighted years with sentiment
    train_by_date = train.groupby(['date']).subjectivity.sum().reset_index()

    highlight1_by_date = highlight1.groupby(['date']).subjectivity.sum().reset_index()

    highlight2_by_date = highlight2.groupby(['date']).subjectivity.sum().reset_index()

    test_by_date = test.groupby(['date']).subjectivity.sum().reset_index()

    # Graph dates
    time_graph = plt.figure(figsize = (16, 10))
    plt.title("Subjectivity Line Graph")
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Summation of Subjectivity Score by date', fontsize=20)
    plt.plot(train_by_date.date, train_by_date.subjectivity)
    plt.plot(highlight1_by_date.date, highlight1_by_date.subjectivity)
    plt.plot(highlight2_by_date.date, highlight2_by_date.subjectivity)
    plt.plot(test_by_date.date, test_by_date.subjectivity)
    plt.show()

# polarity line graph for time series DF
def time_series_graph_polarity(df):
    train = df[df.year == 1999]
    highlight1 = df[df.year == 2000]
    highlight2 = df[df.year == 2001]
    test = df[df.year == 2002]

    # show  highlighted years with sentiment
    train_by_date = train.groupby(['date']).polarity.sum().reset_index()

    highlight1_by_date = highlight1.groupby(['date']).polarity.sum().reset_index()

    highlight2_by_date = highlight2.groupby(['date']).polarity.sum().reset_index()

    test_by_date = test.groupby(['date']).polarity.sum().reset_index()

    # Graph dates
    time_graph = plt.figure(figsize = (16, 10))
    plt.title("Polarity Line Graph")
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('Summation of Polarity Score by date', fontsize=20)
    plt.plot(train_by_date.date, train_by_date.polarity)
    plt.plot(highlight1_by_date.date, highlight1_by_date.polarity)
    plt.plot(highlight2_by_date.date, highlight2_by_date.polarity)
    plt.plot(test_by_date.date, test_by_date.polarity)
    plt.show()



# function that creates seasonality graph using dataframe and requested value for sentiment
def seasonal_plot_sentiment(df):
    # sets y variable
    y = df.sentiment

    #A seasonal subseries plot shows the change year-over-year within each month.
    table = y.groupby([y.index.year, y.index.month]).mean().unstack()

    # Plot table
    fig, axs = plt.subplots(1, 12, sharey=True, sharex=True, figsize=(16,10))
    for ax, (month, subset) in zip(axs, table.iteritems()):
        subset.plot(ax=ax, title=month)
        x_left, x_right = ax.get_xlim()
        ax.hlines(subset.mean(), x_left, x_right, ls='--')
        ax.set(xlabel='')
        
    fig.suptitle('Seasonal Subseries Plot') # super-title for the overall figure
    fig.subplots_adjust(wspace=0)


# function that creates seasonality graph using dataframe and requested value for sentiment
def seasonal_plot_subjectivity(df):
    # sets y variable
    y = df.subjectivity

    #A seasonal subseries plot shows the change year-over-year within each month.
    table = y.groupby([y.index.year, y.index.month]).mean().unstack()

    # Plot table
    fig, axs = plt.subplots(1, 12, sharey=True, sharex=True, figsize=(16,10))
    for ax, (month, subset) in zip(axs, table.iteritems()):
        subset.plot(ax=ax, title=month)
        x_left, x_right = ax.get_xlim()
        ax.hlines(subset.mean(), x_left, x_right, ls='--')
        ax.set(xlabel='')
        
    fig.suptitle('Seasonal Subseries Plot') # super-title for the overall figure
    fig.subplots_adjust(wspace=0)



# function that creates seasonality graph using dataframe and requested value for sentiment
def seasonal_plot_polarity(df):
    # sets y variable
    y = df.polarity

    #A seasonal subseries plot shows the change year-over-year within each month.
    table = y.groupby([y.index.year, y.index.month]).mean().unstack()

    # Plot table
    fig, axs = plt.subplots(1, 12, sharey=True, sharex=True, figsize=(16,10))
    for ax, (month, subset) in zip(axs, table.iteritems()):
        subset.plot(ax=ax, title=month)
        x_left, x_right = ax.get_xlim()
        ax.hlines(subset.mean(), x_left, x_right, ls='--')
        ax.set(xlabel='')
        
    fig.suptitle('Seasonal Subseries Plot') # super-title for the overall figure
    fig.subplots_adjust(wspace=0)

# Do all the graphs for line charts
def chart_time_graphs(df):
    return time_series_graph_sentiment(df), time_series_graph_subjectivity(df), time_series_graph_polarity(df)


# Do all the seasonal Graphs for all values
def chart_seasonal_graphs(df):
    return seasonal_plot_sentiment(df), seasonal_plot_subjectivity(df), seasonal_plot_polarity(df)

