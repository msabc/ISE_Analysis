import pandas as pd

def convert_columns_to_numeric(dataFrame: pd.DataFrame):
    for column in dataFrame:
        dataFrame[column] = pd.to_numeric(dataFrame[column])
        