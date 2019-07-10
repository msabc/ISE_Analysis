# ISE_Analysis
Data set analysis of the *Istanbul Stock Exchange* dataset for a machine learning course.

## To Run
Clone the project and run **python analysis.py**.

## Configure
2 constants defined inside **analysis.py** are configurable:
1) plotting_enabled (*bool*) - True if plots should be displayed, false otherwise.
2) outlier_removal_enabled (*bool*) - True if outliers are to be removed from the original dataset, false otherwise.

### About
Data file is **data_ise.csv**.
Main python script is **analysis.py** which uses functions defined in **helpers.py**.

### Dataset

| Column Name | Description                                       | Data Type |
|-------------|---------------------------------------------------|-----------|
| date        | Date of recorded price                            | Date      |
| ise         | USD based price                                   | Numeric   |
| ise         | Turkish Lira based price                          | Numeric   |
| sp          | S&P 500 Index (New York Stock Exchange)           | Numeric   |
| dax         | Deutscher Aktien Index (Frankfurt Stock Exchange) | Numeric   |
| ftse        | FTSE 100 Index (London Stock Exchange)            | Numeric   |
| nikkei      | Nikkei Index (Tokyo Stock Exchange)               | Numeric   |
| bovespa     | Bovespa Index (Brasil Sao Paulo Stock Exchange)   | Numeric   |
| eu          | MSCI Europe Index                                 | Numeric   |
| em          | MSCI Emerging Markets Index                       | Numeric   |
