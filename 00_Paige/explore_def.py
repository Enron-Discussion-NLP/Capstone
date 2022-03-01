import numpy as np
from matplotlib import pyplot as plt


# A function that creates a time_series line graph for years 1999-2002 and respective values
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

# function that creates seasonality graph using dataframe and requested value
def seasonal_plot(df, x):
    # sets y variable
    y = df.x

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
    return fig, axs