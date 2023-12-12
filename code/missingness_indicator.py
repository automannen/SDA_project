# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# # Suppress SettingWithCopyWarning
# pd.options.mode.chained_assignment = None  # default='warn'

# # Load datasets
# life_expectancy_df = pd.read_csv('../data/life_expectancy.csv')
# pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')

# # Filter data for the year 2014
# filtered_life_expectancy_df = life_expectancy_df[
#   (life_expectancy_df['Variable'].isin(['Females at age 40', 'Males at age 40'])) &
#   (life_expectancy_df['Measure'] == 'Years') &
#   (life_expectancy_df['Year'] == 2014)
# ]
# filtered_pharma_sales_df = pharma_sales_df[
#   (pharma_sales_df['Measure'] == '/capita, US$ purchasing power parity') &
#   (pharma_sales_df['Year'] == 2014)
# ]

# # Drop unnecessary columns
# columns_to_drop = ['VAR', 'UNIT', 'Measure', 'COU', 'YEA', 'Year', 'Flag Codes', 'Flags']
# filtered_life_expectancy_df.drop(columns=columns_to_drop, inplace=True)
# filtered_pharma_sales_df.drop(columns=columns_to_drop, inplace=True)

# # Extract unique countries from both dataframes
# life_expectancy_countries = set(filtered_life_expectancy_df['Country'].unique())
# pharma_sales_countries = set(filtered_pharma_sales_df['Country'].unique())

# countries_in_both = life_expectancy_countries.intersection(pharma_sales_countries)

# # Filter the life expectancy DataFrame to include only common countries
# filtered_life_expectancy_df = filtered_life_expectancy_df[filtered_life_expectancy_df['Country'].isin(countries_in_both)]
# # Filter the pharma sales DataFrame to include only common countries
# filtered_pharma_sales_df = filtered_pharma_sales_df[filtered_pharma_sales_df['Country'].isin(countries_in_both)]


# for life_exp_variable in filtered_life_expectancy_df['Variable'].unique():
#   print(f"\nResults for Life Expectancy: {life_exp_variable}")

#   # Drop unnecessary columns
#   current_life_expectancy_df = filtered_life_expectancy_df[filtered_life_expectancy_df['Variable'] == life_exp_variable]
#   current_life_expectancy_df.drop(columns=['Variable'], inplace=True)

#   # Merge datasets
#   # I want to take missing data in to account
#   # if a whole row is missing for a country and a drug, i want to take that into account

#   merged_df = current_life_expectancy_df.merge(filtered_pharma_sales_df, on=['Country'], how='outer')

#   merged_df.rename(columns={'Value_x': 'Life_Expectancy_Value', 'Variable': 'Pharma_Sales_Variable',
#                             'Value_y': 'Pharma_Sales_Value'}, inplace=True)


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm

# Suppress SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # default='warn'

# Load datasets
life_expectancy_df = pd.read_csv('../data/life_expectancy.csv')
pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')

# Filter data for the year 2014
filtered_life_expectancy_df = life_expectancy_df[
  (life_expectancy_df['Variable'].isin(['Females at age 40', 'Males at age 40'])) &
  (life_expectancy_df['Measure'] == 'Years') &
  (life_expectancy_df['Year'] == 2014)
]
filtered_pharma_sales_df = pharma_sales_df[
  (pharma_sales_df['Measure'] == '/capita, US$ purchasing power parity') &
  (pharma_sales_df['Year'] == 2014)
]

# Drop unnecessary columns
columns_to_drop = ['VAR', 'UNIT', 'Measure', 'COU', 'YEA', 'Year', 'Flag Codes', 'Flags']
filtered_life_expectancy_df.drop(columns=columns_to_drop, inplace=True)
filtered_pharma_sales_df.drop(columns=columns_to_drop, inplace=True)

# Rename columns for clarity
filtered_life_expectancy_df.rename(columns={'Value': 'Life_Expectancy_Value', 'Variable': 'Life_Expectancy_Variable'}, inplace=True)
filtered_pharma_sales_df.rename(columns={'Value': 'Pharma_Sales_Value', 'Variable': 'Pharma_Sales_Variable'}, inplace=True)

# Identify common countries
common_countries = set(filtered_life_expectancy_df['Country']).intersection(set(filtered_pharma_sales_df['Country']))

print(common_countries)

# Filter the dataframes to include only common countries
filtered_life_expectancy_df = filtered_life_expectancy_df[filtered_life_expectancy_df['Country'].isin(common_countries)]
filtered_pharma_sales_df = filtered_pharma_sales_df[filtered_pharma_sales_df['Country'].isin(common_countries)]

# Generate all combinations of countries, Pharma_Sales_Variables
all_combinations = pd.MultiIndex.from_product(
  [common_countries, filtered_pharma_sales_df['Pharma_Sales_Variable'].unique()],
  names=['Country', 'Pharma_Sales_Variable']
)

# Reindex the filtered Pharma Sales Data
filtered_pharma_sales_df.set_index(['Country', 'Pharma_Sales_Variable'], inplace=True)
pharma_sales_reindexed = filtered_pharma_sales_df.reindex(all_combinations, fill_value=np.nan).reset_index()

# Merge with Life Expectancy Data
merged_df = pharma_sales_reindexed.merge(filtered_life_expectancy_df, on=['Country'], how='outer')

# Create missingness indicators and interaction terms
merged_df['Missingness_Indicator'] = merged_df['Pharma_Sales_Value'].isnull().astype(int)
print(merged_df.shape)
copy = merged_df.copy()

# Scale set with a factor of 10 (for random noice while the true values stay constant)
for i in range(49):
  merged_df = merged_df._append(copy)

print(merged_df.shape)

for i, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
  pharma_sales_value = row['Pharma_Sales_Value']

  if not pd.isnull(pharma_sales_value):
    continue

  else:
    pharma_sales_variable = row['Pharma_Sales_Variable']
    life_expectancy_variable = row['Life_Expectancy_Variable']
    # Filter the DataFrame to include only the possible values with the same Pharma_Sales_Variable and Life_Expectancy_Variable
    possible_varables = merged_df[(merged_df['Pharma_Sales_Variable'] == pharma_sales_variable) & (merged_df['Life_Expectancy_Variable'] == life_expectancy_variable)]['Pharma_Sales_Value'].dropna()

    if possible_varables.shape[0] == 0:
      print('No possible values')
      continue

    else:
      # Fill the missing value with a random value from the possible values
      merged_df.loc[i, 'Pharma_Sales_Value'] = possible_varables.sample(n=1).iloc[0]


print(merged_df.columns)

# Save the merged DataFrame to a CSV file
merged_df.to_csv('merged_data.csv', index=False)









