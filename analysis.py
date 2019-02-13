import pandas as pd
import matplotlib.pyplot as plt
import utils as ut
from scipy.stats import kurtosis, skew
#from matplotlib.pyplot import figure

import pylab

# configuring plotting options
plt.style.use('ggplot')
#figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

# removing date from dataframe because we may not need it
mainDataFrame = pd.read_csv("data_ise.csv",skiprows = [0], sep=",", usecols = [1,2,3,4,5,6,7,8,9])

# renaming the USD and TL columns
mainDataFrame.columns = ["TLira_ISE", "USD_ISE", "NewYork_SP500", "DAX", "FTSE", "NIKKEI", "BOVESPA","EU", "EM"]

print("State before normalization: ")
dfshape = mainDataFrame.shape
print("Rows: " + str(dfshape[0]))
print("Columns: " + str(dfshape[1]))
print()

print(mainDataFrame.info())

#print(mainDataFrame.columns[0] > 100)
print()

# replacing all 0's with 'missing'
mainDataFrame = mainDataFrame.replace({
        'NewYork_SP500': {
            0:  "missing",
        },
         'DAX': {
            0:  "missing",
        },
         'FTSE': {
            0:  "missing",
        },
         'NIKKEI': {
            0:  "missing",
        },
         'BOVESPA': {
            0:  "missing",
        },
          'EU': {
            0:  "missing",
        },
          'EM': {
            0:  "missing",
        }
    })

# not including the first two columns and the last column, since they dont have any missing values
mainDataFrame = mainDataFrame[(mainDataFrame['NewYork_SP500'] != 'missing') & (mainDataFrame['DAX'] != 'missing') & (mainDataFrame['FTSE'] != 'missing') & (mainDataFrame['NIKKEI'] != 'missing') & (mainDataFrame['BOVESPA'] != 'missing') & (mainDataFrame['EU'] != 'missing')]

print("State after removing rows with missing values: ")
print(mainDataFrame.shape)
print()

print("*********** STATISTICS: *************")
print("*Feature: NewYork_SP500*")
print()
print(mainDataFrame['NewYork_SP500'].describe())
print()
print(mainDataFrame['DAX'].describe())
print()
print(mainDataFrame['FTSE'].describe())
print()
print(mainDataFrame['NIKKEI'].describe())
print()
print(mainDataFrame['BOVESPA'].describe())
print()
print(mainDataFrame['EU'].describe())
print()
print(mainDataFrame['EM'].describe())
print()

# some series are interpreted as type object
# so we re using a utility method to convert them all to float64's
ut.convert_columns_to_numeric(mainDataFrame)
print()
print("*********** PLOTTING: *************")
print()
print("*********** BOX PLOT: *************")
print("*Feature: NewYork_SP500*")
# boxplot
mainDataFrame.boxplot(figsize=(11,7))
print()

# histogram
# print("***********HISTOGRAM: *************")
# ut.histogram_foreach_column(mainDataFrame)
# pausing the thread to make sure the plots paint themselves
# before the correlation calculation
plt.pause(1)

# scatter
print("*********** SCATTER PLOTS: **************")
pylab.scatter(mainDataFrame.NewYork_SP500.index, mainDataFrame.NewYork_SP500)
pylab.scatter(mainDataFrame.FTSE.index, mainDataFrame.FTSE)
pylab.scatter(mainDataFrame.EM.index, mainDataFrame.EM)
    
# pausing the thread to make sure the plots paint themselves
# before the correlation calculation
plt.pause(1)

print("*********** CORRELATION: **************")
print("*Feature: NewYork_SP500*")
print()
correlation_Array = ut.get_correlation_with_other_columns(mainDataFrame.NewYork_SP500, mainDataFrame)

# sorting the array ASC by the correlation value
correlation_Array.sort(key=lambda tup: tup[1], reverse = True)

for item in correlation_Array:
    print("Variable: {column}, correlation: {correlationAmount}".format(column = item[0], correlationAmount = item[1]))

print()

# check is the payoff of removing outliers and NaN's worth it
#newFrame = ut.remove_outliers(mainDataFrame)
#print(newFrame.head(5))
#print(newFrame.tail(5))

print("*********** SKEWNESS: **************")
print("*Feature: NewYork_SP500*")
print("ALPHA3 equals: " + str(skew(mainDataFrame.NewYork_SP500)) + " => distribution is approximately symmetric")
print()

print("*********** KURTOSIS: **************")
print("*Feature: NewYork_SP500*")
print("ALPHA4 equals: " + str(kurtosis(mainDataFrame.NewYork_SP500)) + " => distribution is very close to the Bell curve")
print()

# ut.calc_iv(mainDataFrame, 'FTSE', 'NewYork_SP500', 1)