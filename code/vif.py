import pandas as pd 
from sklearn.linear_model import LinearRegression

pharma_sales = pd.read_csv('../data/pharma_sales_ppp.csv')

R03 = 'R03-Drugs for obstructive airway diseases'
anti_dep = 'N06A-Antidepressants'
ulcer = 'A02B-Drugs for peptic ulcer and gastro-oesophageal reflux diseases (GORD)'

# All independent variables 
R03_data = pharma_sales[pharma_sales['Variable'] == R03]['Value'].reset_index(drop=True)
anti_depressant_data = pharma_sales[pharma_sales['Variable'] == anti_dep]['Value'].reset_index(drop=True)
ulcer_data = pharma_sales[pharma_sales['Variable'] == ulcer]['Value'].reset_index(drop=True)

# Creating dataframe of all indpendent values data
column_names = ['R03', 'Anti_depressant', 'Peptic ulcer']
data = {
    'R03': R03_data,
    'Anti_depressant': anti_depressant_data,
    'Peptic ulcer': ulcer_data
}

X = pd.DataFrame(data, columns=column_names)

for i, col in enumerate(X.columns):
    y = X[col]
    X_without_col = X.drop(col, axis=1)
    model = LinearRegression().fit(X_without_col, y)
    
    r_squared = model.score(X_without_col, y)
    vif = 1 / (1 - r_squared)
    
    print(f'{col}: {vif}')
    

