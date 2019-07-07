import pandas as pd
import numpy as np
import matplotlib.cm as cm
from scipy.stats import zscore

CONST_bin_number = 'Bin Number'
CONST_events = 'Events'
CONST_non_events = 'Non Events'
CONST_name = 'name'
CONST_group = 'group'
CONST_event_y_n = 'Event (Yes/No)'
CONST_pct_non_events = '% Non Events'
CONST_pct_events = '% Events'
CONST_woe = 'WOE'
CONST_iv = 'IV'
CONST_y = 'Yes'
CONST_n = 'No'

# General
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

# Outliers    
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

# Z Score
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
        zscore_groups.append({
                CONST_name: dataframe_series_names[i], 
                CONST_group: group_zscores(zscore_data_array[i])
                })
    return zscore_groups

# Information Value and Weight of Evidence
def generate_iv_df(s1: pd.Series, s2: pd.Series):
    newDf = create_intermediary_iv_df(s1, s2)
    bin_num = newDf[CONST_bin_number]
    non_events = newDf[CONST_non_events]
    events = newDf[CONST_events]
    total_non_events = non_events.sum()
    total_events = events.sum()
    
    iter_count = len(bin_num.values)
    
    pct_non_events = []
    pct_events = []
    woe = []
    iv = []
    total_iv = 0
    for i in range(0, iter_count):
        curr_non_event_num = non_events.values[i]
        pct_non_event = curr_non_event_num / total_non_events
        pct_non_events.append(pct_non_event)
        
        curr_event_num = events.values[i]
        pct_event = curr_event_num / total_events
        pct_events.append(pct_event)
        
        woe_val = np.log(pct_non_event / pct_event)
        woe.append(woe_val)
        
        iv_val = (pct_non_event - pct_event) * woe_val
        iv.append(iv_val)
        
        total_iv += iv_val
    
    series_pct_non_events = pd.Series(pct_non_events)
    series_pct_events = pd.Series(pct_events)
    series_woe = pd.Series(woe)
    series_iv = pd.Series(iv)
    
    frame = { 
            CONST_bin_number: bin_num, 
            CONST_non_events: non_events, 
            CONST_events: events, 
            CONST_pct_non_events: series_pct_non_events,
            CONST_pct_events: series_pct_events,
            CONST_woe: series_woe,
            CONST_iv: series_iv
            } 
    
    iv_dataFrame = pd.DataFrame(frame)
    return (iv_dataFrame, total_iv)

def create_intermediary_iv_df(s1: pd.Series, s2: pd.Series):
    binned_tuple_collection = get_binned_tuple_of_series(s1, s2)
    df = convert_binned_tuple_collection_to_DataFrame(binned_tuple_collection)
    num_of_bins = len(df[CONST_bin_number].unique())
    
    bins = []
    non_events = []
    events = []
    for i in range (0, num_of_bins):
        bool_extract_bin_number = df[CONST_bin_number] == i
        newDf = df[bool_extract_bin_number]
        total_num_rows = newDf.shape[0]
        churn_number = len(newDf[CONST_event_y_n].where(lambda x : x == CONST_y).dropna())
        non_churn_number = total_num_rows - churn_number
        
        bins.append(i)
        non_events.append(non_churn_number)
        events.append(churn_number)
        
    bin_series = pd.Series(bins)
    non_events_series = pd.Series(non_events)
    events_series = pd.Series(events)
    
    frame = { 
            CONST_bin_number: bin_series, 
            CONST_non_events: non_events_series, 
            CONST_events: events_series} 
    
    return pd.DataFrame(frame)

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
    if(len(bins_s1[len(bins_s1) - 1]) != len(bins_s2[len(bins_s2) - 1])):
        bins_s1.pop()
        bins_s2.pop()
    
    number_of_bins = len(bins_s1)
    tuple_collection = []
    
    for i in range(0, number_of_bins):
        for j in range(0, len(bins_s1[i])):
            bin_number = i
            first_bin_value = bins_s1[i][j]
            second_bin_value = bins_s2[i][j]
            
            if((first_bin_value > 0 and second_bin_value > 0) or 
               (first_bin_value < 0 and second_bin_value < 0)):
                value_tuple = (bin_number, first_bin_value, second_bin_value, CONST_y)
            else:
                value_tuple = (bin_number, first_bin_value, second_bin_value, CONST_n)
            tuple_collection.append(value_tuple)
    
    return tuple_collection



def get_bins(arr, number_of_bins):
    return list(__yield_bins(arr, number_of_bins))

def __yield_bins(arr, number_of_bins):
    bin_size = len(arr) // number_of_bins
    for i in range(0, len(arr), bin_size):
        yield arr[i:i + bin_size]
    
# If anything is changed in the algorithm that creates the binned_tuple_collection,
# those changes need to be reflected here also.
def convert_binned_tuple_collection_to_DataFrame(binned_tuple_collection):
    bin_number_values = []
    s1_values = []
    s2_values = []
    churn_nonchurn_values = []
    
    for i in range(0, len(binned_tuple_collection)):
        bin_number_values.append(binned_tuple_collection[i][0])
        s1_values.append(binned_tuple_collection[i][1])
        s2_values.append(binned_tuple_collection[i][2])
        churn_nonchurn_values.append(binned_tuple_collection[i][3])
    
    bin_num_vals = pd.Series(bin_number_values)
    s1 = pd.Series(s1_values)
    s2 = pd.Series(s2_values)
    s3 = pd.Series(churn_nonchurn_values)
    
    frame = { 
            CONST_bin_number: bin_num_vals, 
            'Series_1': s1, 
            'Series_2': s2, 
            CONST_event_y_n: s3
            } 
    return pd.DataFrame(frame) 
    
    
    
    
    
    
    