import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

variables = [
  'N06A-Antidepressants',
  'A02B-Drugs for peptic ulcer and gastro-oesophageal reflux diseases (GORD)',
  # 'R03-Drugs for obstructive airway diseases',
  'N-Nervous system'
]

variable_codes = {
  'N06A-Antidepressants': 'N06A',
  'A02B-Drugs for peptic ulcer and gastro-oesophageal reflux diseases (GORD)': 'A02B',
  'R03-Drugs for obstructive airway diseases': 'R03',
  'N-Nervous system': 'N'
}

oecd_countries = [
  "Australia", "Austria", "Belgium", "Canada", "Chile", "Czechia", "Denmark",
  "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland",
  "Israel", "Italy", "Japan", "Korea", "Latvia", "Lithuania", "Luxembourg", "Mexico",
  "Netherlands", "New Zealand", "Norway", "Poland", "Portugal", "Slovak Republic",
  "Slovenia", "Spain", "Sweden", "Switzerland", "Türkiye", "United Kingdom", "United States"
]

# Base directory for CSV files
base_dir = '../data/'  # Update this to the path where your CSV files are stored

# Read and filter both dataframes
life_expectancy_df = pd.read_csv(os.path.join(base_dir, 'life_expectancy.csv'))
pharmaceutical_sales_df = pd.read_csv(os.path.join(base_dir, 'pharma_sales_ppp.csv'))

# Filter for OECD countries in both dataframes
life_expectancy_df = life_expectancy_df[life_expectancy_df["Country"].isin(oecd_countries)]
pharmaceutical_sales_df = pharmaceutical_sales_df[pharmaceutical_sales_df["Country"].isin(oecd_countries)]

# Filter pharmaceutical sales DataFrame to include only the desired variables
pharmaceutical_sales_df = pharmaceutical_sales_df[pharmaceutical_sales_df['Variable'].isin(variables)]

# Rename 'Variable' columns before merging to differentiate them
life_expectancy_df.rename(columns={'Variable': 'LifeExpectancyVariable'}, inplace=True)
pharmaceutical_sales_df.rename(columns={'Variable': 'PharmaSalesVariable'}, inplace=True)

# Merge the two dataframes on 'Country' and 'Year'
merged_df = pd.merge(life_expectancy_df, pharmaceutical_sales_df, on=['Country', 'Year'], how='outer')

# Find the top 5 years with the most data
most_data_by_year = merged_df.notna().groupby(merged_df['Year']).sum().sum(axis=1)
top_5_years_most_data = most_data_by_year.nlargest(5).index.tolist()

# Filter the DataFrame to include only the top 5 years
filtered_df = merged_df[merged_df['Year'].isin(top_5_years_most_data)]

# Create the directory for saving the heatmaps
plots_directory = 'life_expectancy_pharmaceutical_sales_heatmaps'
os.makedirs(plots_directory, exist_ok=True)

for variable in life_expectancy_df['LifeExpectancyVariable'].unique():
  # Create a subset of the dataframe for this variable
  subset_df = filtered_df[filtered_df['LifeExpectancyVariable'] == variable].copy()  # Use .copy() to explicitly create a copy

  # Replace full variable names with short codes in the 'Country-Variable' column
  subset_df['Country-Variable'] = subset_df['Country'] + ' - ' + subset_df['PharmaSalesVariable'].map(variable_codes)

  # Pivot this subset DataFrame
  pivoted_df = subset_df.pivot_table(index='Year',
                                      columns='Country-Variable',
                                      values='Value_y',  # Assuming 'Value_y' is the correct column for sales data
                                      aggfunc='first')

  country_names = [name.split(" -")[0] for name in pivoted_df.columns]

  # Now get the unique country names
  unique_countries = sorted(set(country_names))

  # print(unique_countries)
  # print(set(oecd_countries) - set(unique_countries))

  # Create and save the heatmap
  plt.figure(figsize=(20, 10))  # Adjust the size as needed
  sns.heatmap(pivoted_df.isna(), cbar=False, linewidths=.1)
  plt.title(f"Missing Values - {variable}", fontsize=16)
  plt.xlabel('Country - Drug Sales Variables', fontsize=14)
  plt.ylabel('Year', fontsize=14)
  plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
  # plt.xticks(fontsize=5)  # Adjust the fontsize as needed
  plt.tight_layout()
  plt.subplots_adjust(bottom=0.2)  # Adjust if needed to prevent xlabel cutoff

  heatmap_path = os.path.join(plots_directory, f'heatmap_{variable}_with_countries_and_variables.png')
  plt.savefig(heatmap_path)
  plt.close()

  print(f"Heatmap saved to {heatmap_path}")
