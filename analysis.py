# The goal of this script is to analyze the stock market data and answer the 
# question how variables 
# Stock market return index of Germany (DAX)
# Stock market return index of UK (FTSE)
# Stock market return index of Japan (NIKKEI)
# Stock market return index of Brazil (BOVESPA)
# MSCI European index (EU)
# MSCI emerging markets index (EM)
# affect the New York Stock Exchange (NewYork_SP500), and also to try
# to predict the value of the New York Stock Exchange for a given 
# combination of the variables above.

import pandas as pd
import matplotlib.pyplot as plt
import helpers as hp
from scipy.stats import kurtosis, skew
import pylab


# constants used throughout the script
# any change here will not affect the execution of the script
NEW_YORK_INDEX = 'NewYork_SP500'
GERMANY_INDEX = 'DAX'
UK_INDEX = 'FTSE'
JAPAN_INDEX = 'NIKKEI'
BRAZIL_INDEX = 'BOVESPA'
EU_INDEX = 'EU'
EM_INDEX = 'EM'

MISSING_CONST = 'missing'

plotting_enabled = True

# configuring plotting options
plt.style.use('ggplot')
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

# removing 3 variables from the data set since they won't be used for calculation:
# [0] date 
# [1] ISE (Turkish Lira) 
# [2] ISE (USD) 
mainDataFrame = pd.read_csv("data_ise.csv",
                            skiprows = [0], 
                            sep=",", 
                            usecols = [3,4,5,6,7,8,9])

# enables giving custom names to columns
mainDataFrame.columns = [NEW_YORK_INDEX, 
                         GERMANY_INDEX, 
                         UK_INDEX, 
                         JAPAN_INDEX, 
                         BRAZIL_INDEX, 
                         EU_INDEX, 
                         EM_INDEX]

print("State before normalization: ")
print("Rows: " + str(mainDataFrame.shape[0]))
print("Columns: " + str(mainDataFrame.shape[1]))
print()

print(mainDataFrame.info())
print("Dependent variable: " + NEW_YORK_INDEX)
print()

# replacing all 0's with 'missing'
mainDataFrame = mainDataFrame.replace({
        NEW_YORK_INDEX: {
            0:  MISSING_CONST
        },
         GERMANY_INDEX: {
            0:  MISSING_CONST
        },
         UK_INDEX: {
            0:  MISSING_CONST
        },
         JAPAN_INDEX: {
            0:  MISSING_CONST
        },
         BRAZIL_INDEX: {
            0:  MISSING_CONST
        },
         EU_INDEX: {
            0:  MISSING_CONST
        },
         EM_INDEX: {
            0:  MISSING_CONST
        }
    })

# Since normalization of a specific value inside a column is rarely used
# I decided to remove the whole rows of data that contained a single 'missing'
# value

rows_before_normalization = mainDataFrame.shape[0]

# not including the first two columns and the last column, 
# since they dont have any missing values
mainDataFrame = mainDataFrame[(mainDataFrame[NEW_YORK_INDEX] != MISSING_CONST) & 
                              (mainDataFrame[GERMANY_INDEX] != MISSING_CONST) & 
                              (mainDataFrame[UK_INDEX] != MISSING_CONST) & 
                              (mainDataFrame[JAPAN_INDEX] != MISSING_CONST) & 
                              (mainDataFrame[BRAZIL_INDEX] != MISSING_CONST) & 
                              (mainDataFrame[EU_INDEX] != MISSING_CONST)]

rows_after_normalization = mainDataFrame.shape[0]
print('Number of rows after normalization: ' + str(rows_after_normalization))
num_of_rows_lost = rows_before_normalization - rows_after_normalization
print('Lost ' + str(num_of_rows_lost) + ' rows')
print('Percentage lost: ' + str((num_of_rows_lost / rows_before_normalization) * 100))
print()

# at this point some series are interpreted as type object 
# even though the result of describe method says float64,
# so we're using a utility method to convert them all to float64's
mainDataFrame = hp.convert_columns_to_numeric(mainDataFrame)

print("*********** STATISTICS: *************")
print("Feature: " + NEW_YORK_INDEX)
print()
print(mainDataFrame[NEW_YORK_INDEX].describe())
print()
print(mainDataFrame[GERMANY_INDEX].describe())
print()
print(mainDataFrame[UK_INDEX].describe())
print()
print(mainDataFrame[JAPAN_INDEX].describe())
print()
print(mainDataFrame[BRAZIL_INDEX].describe())
print()
print(mainDataFrame[EU_INDEX].describe())
print()
print(mainDataFrame[EM_INDEX].describe())
print()

print("Z-score: ")
# if a z-score is positive, its’ corresponding value is greater than the mean 
# (on the right side of the mean)

# if a z-score is negative, its' corresponding value is lower than the mean 
# (on the left side)

# a zscore_groups represents a data of the following format:
# zscore_group[0]['name'] is the name of the first Series
# zscore_group[0]['group'] is a an array of 3 elements of the following format:
# zscore_group[0]['group'][0] are zscores for a specific series with a value less than -3
# zscore_group[0]['group'][1] are zscores for a specific series with a value between -3 and 3
# zscore_group[0]['group'][2] are zscores for a specific series with a value greater than 3
zscore_groups = hp.generate_zscore_groups(mainDataFrame)

for i in range(len(zscore_groups)):
    print(zscore_groups[i]['name'])
    print('Number of values less than -3: ' + str(len(zscore_groups[i]['group'][0])))
    print('Number of values between 3 and -3: ' + str(len(zscore_groups[i]['group'][1])))
    print('Number of values greater than 3: ' + str(len(zscore_groups[i]['group'][2])))
    print()

print()

if plotting_enabled:
    print("*********** PLOTTING: *************")
    print("*********** BOX PLOT: *************")
    print("*Feature: " + NEW_YORK_INDEX)
    
    # boxplot
    mainDataFrame.boxplot(figsize=(11,7))
    print()
    
    # pausing the thread to make sure the plots paint themselves
    # before the correlation calculation
    plt.pause(1)
    
    # histogram
    print("***********HISTOGRAM: *************")
    hp.histogram_foreach_column(mainDataFrame)
    print()
    
    plt.pause(1)
    
    # scatter
    print("*********** SCATTER PLOT **********")
    pylab.scatter(mainDataFrame.NewYork_SP500.index, mainDataFrame.NewYork_SP500)
    pylab.scatter(mainDataFrame.BOVESPA.index, mainDataFrame.BOVESPA)
    pylab.scatter(mainDataFrame.DAX.index, mainDataFrame.DAX)
    pylab.scatter(mainDataFrame.EU.index, mainDataFrame.EU)
        
    # pausing the thread to make sure the plots paint themselves
    # before the correlation calculation
    plt.pause(1)

print("*********** CORRELATION: **************")
print("*Feature: " + NEW_YORK_INDEX)
print()
correlation_Array = hp.get_correlation_with_other_columns(mainDataFrame.NewYork_SP500, mainDataFrame)

# sorting the array ASC by the correlation value
correlation_Array.sort(key=lambda tup: tup[1], reverse = True)

for item in correlation_Array:
    print("Variable: {column}, correlation: {correlationAmount}"
          .format(column = item[0], correlationAmount = item[1]))

print()

# TODO: check if the payoff of removing outliers and NaN's is worth it
# considering we lose a significant amount of data
#mainDataFrame = hp.remove_outliers(mainDataFrame)
#print(mainDataFrame.head(5))
#print(mainDataFrame.tail(5))

print("*********** SKEWNESS: **************")
print("*Feature: " + NEW_YORK_INDEX)
print("ALPHA3 equals: " + str(skew(mainDataFrame.NewYork_SP500)) + 
      " => distribution is approximately symmetric")
print()

print("*********** KURTOSIS: **************")
print("*Feature: " + NEW_YORK_INDEX)
print("ALPHA4 equals: " + str(kurtosis(mainDataFrame.NewYork_SP500)) + 
      " => distribution is very close to the Bell curve")
print()

iv_tuple = hp.generate_iv_df(mainDataFrame[NEW_YORK_INDEX], mainDataFrame[GERMANY_INDEX])
print(iv_tuple[0])
print('Total: ' + str(iv_tuple[1]))