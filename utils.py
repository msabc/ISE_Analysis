import pandas as pd

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