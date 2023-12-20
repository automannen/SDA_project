# vif (variance inflation factor)
# testing for multicolinearity between the independent variables

import pandas as pd
from sklearn.linear_model import LinearRegression

# the chosen countries in our first attempt of the linear model that all contained
# data for the drugs in the year 2014:
# - N-Nervous system
# - N06A-Antidepressants
# - A02B-Drugs for peptic ulcer and gastro-oesphageal reflux diseases (GORD)
chosen_countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Czechia',
                    'Estonia', 'Finland', 'Germany', 'Greece', 'Hungary', 'Iceland',
                    'Ireland', 'Italy', 'Korea', 'Latvia', 'Luxembourg', 'Netherlands',
                    'New Zealand', 'Norway', 'Portugal', 'Slovak Republic', 'Slovenia',
                    'Spain', 'Sweden', 'Türkiye']


pharma_sales = pd.read_csv('../data/pharma_sales_ppp.csv')
pharma_sales_2014 = pharma_sales[pharma_sales['Year'] == 2014]

# The 3 selected drugs
nervous = 'N-Nervous system'
anti_dep = 'N06A-Antidepressants'
ulcer = 'A02B-Drugs for peptic ulcer and gastro-oesophageal reflux diseases (GORD)'

# All independent variables data for 2014 and the chosen countries
nervous_data = pharma_sales_2014[pharma_sales_2014['Variable'] == nervous][['Value', 'Country']].reset_index(drop=True)
nervous_data_filtered = nervous_data[nervous_data['Country'].isin(chosen_countries)].reset_index(drop=True)

anti_depressant_data = pharma_sales_2014[pharma_sales_2014['Variable'] == anti_dep][['Value', 'Country']].reset_index(drop=True)
anti_depressant_data_filtered = anti_depressant_data[anti_depressant_data['Country'].isin(chosen_countries)].reset_index(drop=True)

ulcer_data = pharma_sales_2014[pharma_sales_2014['Variable'] == ulcer][['Value', 'Country']].reset_index(drop=True)
ulcer_data_filtered = ulcer_data[ulcer_data['Country'].isin(chosen_countries)].reset_index(drop=True)

# Creating dataframe of all independent values data
column_names = ['N-nervous system', 'Anti_depressant', 'Peptic ulcer']
data = {column_names[0]: nervous_data_filtered['Value'], column_names[1]: anti_depressant_data_filtered['Value'], column_names[2]: ulcer_data_filtered['Value']}

X = pd.DataFrame(data, columns=column_names)
print(X)

# fitting a linear regression model with different independent variable where
# the dependent variable is a drug and is removed from the set of independent
# variables
for i, col in enumerate(X.columns):
    y = X[col]
    X_without_col = X.drop(col, axis=1)
    model = LinearRegression().fit(X_without_col, y)

    r_squared = model.score(X_without_col, y)
    # the bigger the r squared the higher the vif value
    vif = 1 / (1 - r_squared)

    print(f'{col}: {vif}')


