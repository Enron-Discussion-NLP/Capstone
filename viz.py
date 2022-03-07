import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_1(time_series_df):
    '''
    This function is a univariate exploration of the distribution of the continuous variables.
    '''
    cont_vars = ['intensity', 'polarity', 'subjectivity', 'year']
    bool_vars = ['poi', 'is_internal']

    # distributions of continuous variables
    for var in cont_vars:
        plt.figure(figsize = (10, 4))
        sns.histplot(data = time_series_df, x = var, kde = True)
        plt.title(var)
        plt.show()
        
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_2(df):
    '''
    This function plots the distribution of internal emails
    '''
    # internal vs not internal emails
    plt.figure(figsize = (6, 4))
    sns.countplot(data = df, x = 'is_internal', palette = ['grey', '#912a15'])
    plt.title(f'{round(df.is_internal.mean()*100)}% of Emails Sent Internally')
    plt.show()
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_3(df):
    '''
    This function plots the distribution of observations marked for persons of interest
    '''
    # emails sent from a non-person of of interest vs person of interest
    plt.figure(figsize = (6, 10))
    sns.countplot(data = df, x = 'is_poi', palette = ['#912a15', 'grey'])
    plt.title(f'{round(df.is_poi.mean()*100, 2)}% of Emails Sent from Person of Interest')
    plt.show()
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_4(time_series_df):
    '''
    This function creates boxplots of intensity scores over time
    '''
    # plotting intensity score by year to look at median, IQR, and outliers
    plt.figure(figsize = (10, 6))
    sns.boxenplot(data = time_series_df, x = 'year', y = 'intensity', palette = ['grey', '#d3aaa1', '#b26a5b', '#b26a5b', \
                                                                     '#a75544', '#9c3f2c'])
    plt.title('Intensity  Scores Appear Similar Across Year, IQR Variance Increases Over Time')
    plt.show()
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_5(time_series_df):
    '''
    This function creates boxplots for polarity overtime
    '''
    # plotting polarity score by year to look at median, IQR, and outliers
    plt.figure(figsize = (10, 6))
    sns.boxenplot(data = time_series_df, x = 'year', y = 'polarity', palette = ['grey', '#d3aaa1', '#b26a5b', '#b26a5b', \
                                                                     '#a75544', '#9c3f2c'])
    plt.title('Polarity Scores Appear Similar Across Year, IQR Variance Decreases Over Time')
    plt.show()
    
    
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_6(time_series_df):
    '''
    The function creates boxplots for subjectivity overtime
    '''
    # plotting subjectivity score by year to look at median, IQR, and outliers
    plt.figure(figsize = (10, 6))
    sns.boxenplot(data = time_series_df, x = 'year', y = 'subjectivity', palette = ['grey', '#d3aaa1', '#b26a5b', '#b26a5b', \
                                                                     '#a75544', '#9c3f2c'])
    plt.title('Subjectivity Appears to Descrease Over Time')
    plt.show()



# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_7(df):
    '''
    This function creates boxplots for the distribution of intensity for internal emails
    '''
    # do internal emails have higher or lower sentiment scores than external emails?
    sns.boxenplot(data = df, x = 'is_internal', y = 'intensity', palette = ['grey', '#912a15'])
    plt.title('Internal and External Sentiment about the Same, Internal Emails have Greater Variance')
    plt.show()
    
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_8(df):
    '''
    This function creates boxplots for polarity of the internal emails
    '''
    sns.boxenplot(data = df, x = 'is_internal', y = 'polarity', palette = ['grey', '#912a15'])
    plt.title('Polarity Appears Mostly Similar for Internal and External Emails')
    plt.show()
    
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_9(df):
    '''
    This function creates boxplots for subjectivity for internal emails
    '''
    sns.boxenplot(data = df, x = 'is_internal', y = 'subjectivity', palette = ['grey', '#912a15'])
    plt.title('Mean Subjectivity Appears to be Slightly Lower for Internal Emails')
    plt.show()
    
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_10(df):
    '''
    This function creates boxplots for the distribution of intensity for poi vs non-poi emails
    '''
    # do internal emails have higher or lower sentiment scores than external emails?
    sns.boxenplot(data = df, x = 'is_poi', y = 'intensity', palette = ['grey', '#912a15'])
    plt.title('POI vs Non-POI Intensity about the Same')
    plt.show()
    
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_11(df):
    '''
    This function creates boxplots for polarity for poi vs non-poi emails
    '''
    sns.boxenplot(data = df, x = 'is_poi', y = 'polarity', palette = ['grey', '#912a15'])
    plt.title('POI vs Non-POI Polarity about the Same')
    plt.show()
    
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_12(df):
    '''
    This function creates boxplots for subjectivity for for poi vs non-poi emails
    '''
    sns.boxenplot(data = df, x = 'is_poi', y = 'subjectivity', palette = ['grey', '#912a15'])
    plt.title('POI vs Non-POI Subjectivity about the Same')
    plt.show()
    
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_13(df):
    '''
    this function builds a correlation of the df and then creates a heatmap to find the correlation of 
    the different features
    '''
    # Creat correlation table and heatmap
    corr_table =  df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_table, cmap='Reds', annot=True, linewidth=0.5, mask= np.triu(corr_table))
    plt.title('Sentiment Measures Highly Correlated, Intensity has Highest Correlation with POI')
    plt.show()  
    
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------

def viz_14(time_series_df):
    '''
    This function creates a line plots intesnsity over 1999-2001 that is colored by each year.
    '''
    # creating a ts for each year
    ts_1999 = time_series_df[time_series_df.year == 1999]
    ts_2000 = time_series_df[time_series_df.year == 2000]
    ts_2001 = time_series_df[time_series_df.year == 2001]
    ts_2002 = time_series_df[time_series_df.year == 2002]

    # creating dfs grouped by date for plotting
    ts_1999_grouped_intensity = ts_1999.groupby('date').intensity.sum().reset_index()
    ts_2000_grouped_intensity = ts_2000.groupby('date').intensity.sum().reset_index()
    ts_2001_grouped_intensity = ts_2001.groupby('date').intensity.sum().reset_index()
    ts_2002_grouped_intensity = ts_2002.groupby('date').intensity.sum().reset_index()

    # plotting insensity score by date
    plt.figure(figsize = (16, 6))

    plt.plot(ts_1999_grouped_intensity.date, ts_1999_grouped_intensity.intensity, color = 'grey')
    plt.plot(ts_2000_grouped_intensity.date, ts_2000_grouped_intensity.intensity, color = '#299e5bff')
    plt.plot(ts_2001_grouped_intensity.date, ts_2001_grouped_intensity.intensity, color = '#e92532ff')
    plt.plot(ts_2002_grouped_intensity.date, ts_2002_grouped_intensity.intensity, color = '#0084c7ff')

    plt.title('2000 and 2001 Intense Years for Enron')
    plt.show()
    
# ---------------------------------------------------------------
# ---------------------------------------------------------------   
    
def viz_15(time_series_df):
    '''
    The function creates a seasonal plot that identifies pattern by the month
    '''
    # seasonal subseries plot shows the change year-over-year within each month.
    table = time_series_df.groupby([time_series_df.index.year, time_series_df.index.month]).mean().unstack()

    # getting intensity series for seasonal plot
    y = time_series_df.intensity

    #A seasonal subseries plot shows the change year-over-year within each month.
    table = y.groupby([y.index.year, y.index.month]).mean().unstack()

    #plotting seasonal chart
    fig, axs = plt.subplots(1, 12, sharey=True, sharex=True, figsize=(16,10))
    for ax, (month, subset) in zip(axs, table.iteritems()):
        subset.plot(ax=ax, title=month)
        x_left, x_right = ax.get_xlim()
        ax.hlines(subset.mean(), x_left, x_right, ls='--')
        ax.set(xlabel='')

    fig.suptitle('Intensity by Month Tanks in July, Sept, and Oct, Dec Shoots Up') # super-title for the overall figure
    fig.subplots_adjust(wspace=0)
    
    
    
    
    


    
