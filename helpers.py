import pandas as pd
import numpy as np
import matplotlib.cm as cm
from scipy.stats import zscore

def generate_colors(n:int):
    colors = cm.rainbow(np.linspace(0, 1, n))
    return colors

def convert_columns_to_numeric(dataFrame: pd.DataFrame):
    newDataFrame = dataFrame
    for column in newDataFrame:
        newDataFrame[column] = pd.to_numeric(newDataFrame[column])
    return newDataFrame
        
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
        # pearson by default
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

def get_binned_tuple_of_series(s1: pd.Series, s2: pd.Series):
    num_of_bins = 10
    bins_s1 = get_bins(s1.values, num_of_bins)
    bins_s2 = get_bins(s2.values, num_of_bins)
    
    # raise an error if the number of bins is not equal between Series
    if(len(bins_s1) != len(bins_s2)):
        raise ValueError('Number of bins between Series is different.')
    
    # check if the number of data inside the last bin
    # is different between the Series: 
    # if true remove last bin (unoptimized)
    if(bins_s1[len(bins_s1) - 1] != bins_s2[len(bins_s2) - 1]):
        bins_s1.pop()
        bins_s2.pop()
    
    number_of_bins = len(bins_s1)
    
    tuple_collection = []
    for i in range(0, number_of_bins):
        for j in range(0, bins_s1):
            first_bin_value = bins_s1[i][j]
            second_bin_value = bins_s2[i][j]
            
            if((first_bin_value > 0 and second_bin_value > 0) or (first_bin_value < 0 and second_bin_value < 0)):
                value_tuple = (first_bin_value, second_bin_value, 'Yes')
            else:
                value_tuple = (first_bin_value, second_bin_value, 'No')
            tuple_collection.append(value_tuple)
    
    return tuple_collection

def group_zscores(zscore_arr):
    less_than_minus_three = []
    between_minus_three_and_three = []
    greater_than_three = []
    for i in zscore_arr:
        if (i < -3):
            less_than_minus_three.append(i)
        elif (i > -3 and i < 3):
            between_minus_three_and_three.append(i)
        else:
            greater_than_three.append(i)
    return (less_than_minus_three, between_minus_three_and_three, greater_than_three)

def generate_zscore_groups(dataFrame: pd.DataFrame):
    zscore_data_array = []
    zscore_groups = []
    dataframe_series_names = []
    for column in dataFrame:
        zscore_data_array.append(zscore(dataFrame[column]))
        dataframe_series_names.append(column)
    for i in range(len(zscore_data_array)):
        zscore_groups.append({'name': dataframe_series_names[i], 'group': group_zscores(zscore_data_array[i])})
    return zscore_groups

def get_bins(arr, number_of_bins):
    return list(__yield_bins(arr, number_of_bins))

def __yield_bins(arr, number_of_bins):
    bin_size = len(arr) // number_of_bins
    for i in range(0, len(arr), bin_size):
        yield arr[i:i + bin_size]
    
    
    
    
    
    
    
    
    
    
    
    
    