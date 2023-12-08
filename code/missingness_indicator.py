import pandas as pd
import statsmodels.api as sm

chosen_countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Czechia', 'Estonia', 'Finland', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Korea', 'Latvia', 'Luxembourg', 'Netherlands', 'New Zealand', 'Norway', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden', 'Türkiye']

import pandas as pd

# Load the life_expectancy data
life_expectancy_df = pd.read_csv('../data/life_expectancy.csv')

# Filter life_expectancy data
filtered_life_expectancy_df = life_expectancy_df[
    (life_expectancy_df['Variable'].isin(['Females at age 40', 'Males at age 40'])) &
    (life_expectancy_df['Measure'] == 'Years') &
    (life_expectancy_df['Year'] == 2014)
]

# Load the pharma_sales_ppp data
pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')

# Filter pharma_sales_ppp data for the year 2014
filtered_pharma_sales_df = pharma_sales_df[pharma_sales_df['Year'] == 2014]


merged_df = filtered_life_expectancy_df.merge(filtered_pharma_sales_df, on=['Country', 'Year'], how='outer')

# Assuming you want to drop the column named 'Column_Name_to_Drop'
merged_df = merged_df.drop(columns=['VAR_x', "VAR_y", 'UNIT_x', 'UNIT_y', 'Year', 'COU_x', 'COU_y', 'YEA_x', 'YEA_y', 'Flags_x', 'Flags_y', 'Flag Codes_x', 'Flag Codes_y'])

print(merged_df.columns)

# TODO: onderstaande voorbeeld code toepassen op de merged_df

# # Create a missing indicator variable for 'Income'
# df['Income_Missing'] = df['Income'].isnull().astype(int)

# # Define your model with the indicator variable
# X = df[['Age', 'Income', 'Income_Missing']]
# y = df['Happiness']

# # Add a constant term to the model (intercept)
# X = sm.add_constant(X)

# # Fit the regression model
# model = sm.OLS(y, X, missing='drop')  # 'missing' parameter drops rows with missing values
# results = model.fit()

# # Print the summary statistics
# print(results.summary())
