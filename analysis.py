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