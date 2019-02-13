import pandas as pd
import numpy as np
import matplotlib.cm as cm

def generate_colors(n:int):
    colors = cm.rainbow(np.linspace(0, 1, n))
    return colors

def convert_columns_to_numeric(dataFrame: pd.DataFrame):
    for column in dataFrame:
        dataFrame[column] = pd.to_numeric(dataFrame[column])
        
def histogram_foreach_column(dataFrame: pd.DataFrame, size = (10,7)):
    for column in dataFrame:
        dataFrame.hist(figsize=size, column = column)    
        
def get_types(dataFrame: pd.DataFrame):
    for column in dataFrame:
        print(column.dtype)
        
def get_correlation_with_other_columns(correlationSubject: pd.Series, dataFrame: pd.DataFrame):
    correlation_Array = []
    for col in dataFrame:
        if col == correlationSubject.name:
            continue
        correlation = correlationSubject.corr(other = dataFrame[col])
        correlation_Array.append((col, correlation))
    return correlation_Array

def get_N_largest_numbers(arr, n):
    final_list = [] 
    for i in range(0, n):  
        max1 = 0
        for j in range(len(arr)):      
            if arr[j] > max1: 
                max1 = arr[j] 
        arr.remove(max1) 
        final_list.append(max1) 
        
def get_outliers(data_1):
    outliers=[]
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

def remove_outliers(dataFrame: pd.DataFrame):
    newDataFrame = pd.DataFrame()
    columnIndexer = 0
    for column in dataFrame:
        col = dataFrame[column]
        seriesName = col.name
        col_value_array = col.values.flatten().tolist()
        for outlier in get_outliers(col_value_array):
            col_value_array.remove(outlier)
        newSeries = pd.Series(data = col_value_array, name = seriesName)
        newDataFrame.insert(loc = columnIndexer, value = newSeries, column = newSeries.name)
        columnIndexer += 1
    return newDataFrame

def calc_iv(df, feature: str, target: str, print_output=0):
    
    if df.empty or not feature or not target:
        return None
    
    lst = []

    # iterating over distinct values of a feature column
    for i in range(df[feature].nunique()):
        # getting the specific value
        val = list(df[feature].unique())[i]
        lst.append([feature, val, df[df[feature] == val].count()[feature], df[(df[feature] == val) & (df[target] == 1)].count()[feature]])

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Bad'])
    data = data[data['Bad'] > 0]

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    data['IV'] = (data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])).sum()

    data = data.sort_values(by=['Variable', 'Value'], ascending=True)

    if print_output == 1:
        print(data)