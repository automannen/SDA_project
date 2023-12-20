import pandas as pd
import numpy as np
from tqdm import tqdm

# Suppress SettingWithCopyWarning for cleaner output
pd.options.mode.chained_assignment = None

# Load datasets
life_expectancy_df = pd.read_csv('../data/life_expectancy.csv')
pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')

# Filter data for the year 2014 and specific measures
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
rename_dict_life = {'Value': 'Life_Expectancy_Value', 'Variable': 'Life_Expectancy_Variable'}
rename_dict_pharma = {'Value': 'Pharma_Sales_Value', 'Variable': 'Pharma_Sales_Variable'}
filtered_life_expectancy_df.rename(columns=rename_dict_life, inplace=True)
filtered_pharma_sales_df.rename(columns=rename_dict_pharma, inplace=True)

# Identify common countries
common_countries = set(filtered_life_expectancy_df['Country']).intersection(
    set(filtered_pharma_sales_df['Country'])
)

# Filter the dataframes to include only common countries
filtered_life_expectancy_df = filtered_life_expectancy_df[
    filtered_life_expectancy_df['Country'].isin(common_countries)
]
filtered_pharma_sales_df = filtered_pharma_sales_df[
    filtered_pharma_sales_df['Country'].isin(common_countries)
]

# Generate all combinations of countries and Pharma_Sales_Variables
all_combinations = pd.MultiIndex.from_product(
    [common_countries, filtered_pharma_sales_df['Pharma_Sales_Variable'].unique()],
    names=['Country', 'Pharma_Sales_Variable']
)

# Reindex the filtered Pharma Sales Data
filtered_pharma_sales_df.set_index(['Country', 'Pharma_Sales_Variable'], inplace=True)
pharma_sales_reindexed = filtered_pharma_sales_df.reindex(all_combinations, fill_value=np.nan).reset_index()

# Merge with Life Expectancy Data
merged_df = pharma_sales_reindexed.merge(filtered_life_expectancy_df, on=['Country'], how='outer')

# Create missingness indicators
merged_df['Missingness_Indicator'] = merged_df['Pharma_Sales_Value'].isnull().astype(int)

# Fill missing Pharma Sales Values
for i, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
    if pd.isnull(row['Pharma_Sales_Value']):
        # Filter for possible values with the same Pharma_Sales_Variable and Life_Expectancy_Variable
        possible_values = merged_df[
            (merged_df['Pharma_Sales_Variable'] == row['Pharma_Sales_Variable']) &
            (merged_df['Life_Expectancy_Variable'] == row['Life_Expectancy_Variable'])
        ]['Pharma_Sales_Value'].dropna()

        # Fill missing value with a random value from possible values, if any
        if not possible_values.empty:
            merged_df.at[i, 'Pharma_Sales_Value'] = possible_values.sample(n=1).iloc[0]

# Save the merged DataFrame to a CSV file
merged_df.to_csv('merged_data.csv', index=False)
