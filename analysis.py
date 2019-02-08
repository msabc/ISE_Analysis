import pandas as pd

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

# histogram
mainDataFrame.hist(figsize=(10,7))

print(mainDataFrame.describe())

