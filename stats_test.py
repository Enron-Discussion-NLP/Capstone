# imports
from scipy import stats
from scipy.stats import ttest_ind_from_stats
import numpy as np
α = 0.05



def stats_t_test_1tail_2sample_inten():
    """
    This function do t-test for 1 tail with 2 samples for intensity.
    """
    # creating series for intensitys for 2000 and 2001
    year_2000 = poi_df[poi_df.year_2000 == True].intensity
    year_2001 = poi_df[poi_df.year_2000 == False].intensity
    t, p = stats.ttest_ind(year_2000, year_2001, equal_var= False)

    t, p/2, α
    
    if (p/2 < α) & (t > 0):
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')
        
def stats_t_test_1tail_2sample_senti_pola():
    """
    This function do t-test for 1 tail with 2 samples for polarity.
    """
    # creating series for polaritys for 2000 and 2001
    year_2000 = poi_df[poi_df.year_2000 == True].polarity
    year_2001 = poi_df[poi_df.year_2000 == False].polarity
    t, p = stats.ttest_ind(year_2000, year_2001, equal_var= False)

    t, p/2, α
    
    if (p/2 < α) & (t > 0):
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')
        
def stats_t_test_1tail_2sample_sub():
    """
    This function do t-test for 1 tail with 2 samples for subjectivity.
    """
    # creating series for subjectivitys for 2000 and 2001
    year_2000 = poi_df[poi_df.year_2000 == True].subjectivity
    year_2001 = poi_df[poi_df.year_2000 == False].subjectivity
    t, p = stats.ttest_ind(year_2000, year_2001, equal_var= False)

    t, p/2, α
    
    if (p/2 < α) & (t > 0):
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

