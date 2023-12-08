import pandas as pd 
from sklearn.linear_model import LinearRegression

pharma_sales = pd.read_csv('../data/pharma_sales_ppp.csv')
pharma_sales_2014 = pharma_sales[pharma_sales['Year'] == 2014]
nervous = 'N-Nervous system' 
anti_dep = 'N06A-Antidepressants'
ulcer = 'A02B-Drugs for peptic ulcer and gastro-oesophageal reflux diseases (GORD)'

# All independent variables data for 2014
nervous_data = pharma_sales_2014[pharma_sales_2014['Variable'] == nervous]['Value'][0:28].reset_index(drop=True)
anti_depressant_data = pharma_sales_2014[pharma_sales_2014['Variable'] == anti_dep]['Value'].reset_index(drop=True)
ulcer_data = pharma_sales_2014[pharma_sales_2014['Variable'] == ulcer]['Value'].reset_index(drop=True)

# Creating dataframe of all independent values data
column_names = ['N-nervous system', 'Anti_depressant', 'Peptic ulcer']
data = {column_names[0]: nervous_data, column_names[1]: anti_depressant_data,column_names[2]: ulcer_data}

X = pd.DataFrame(data, columns=column_names)
print(X)

for i, col in enumerate(X.columns):
    y = X[col]
    X_without_col = X.drop(col, axis=1)
    model = LinearRegression().fit(X_without_col, y)
    
    r_squared = model.score(X_without_col, y)
    vif = 1 / (1 - r_squared)
    
    print(f'{col}: {vif}')
    

