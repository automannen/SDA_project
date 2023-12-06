import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define the list of countries and years of interest
countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Czechia',
             'Estonia', 'Finland', 'Germany', 'Greece', 'Hungary', 'Iceland',
             'Ireland', 'Italy', 'Korea', 'Latvia', 'Luxembourg', 'Netherlands',
             'New Zealand', 'Norway', 'Portugal', 'Slovak Republic', 'Slovenia',
             'Spain', 'Sweden', 'Türkiye']
years = [2011, 2012, 2013, 2014, 2016]

# Base directory for CSV files
base_dir = '../data/'  # Replace with the path to your data directory

# Directory for saving plots
plots_directory = 'tobacco_heatmaps'
os.makedirs(plots_directory, exist_ok=True)

# Read the tobacco consumption data
tobacco_consump_df = pd.read_csv(os.path.join(base_dir, 'tobacco_consump.csv'))

# Filter for specified countries and years
tobacco_consump_df = tobacco_consump_df[tobacco_consump_df["Country"].isin(countries) & tobacco_consump_df['Year'].isin(years)]

# Get unique measures
measures = tobacco_consump_df['Measure'].unique()

# Create a heatmap for each measure
for measure in measures:
  # Filter the dataframe for the current measure
  df_filtered = tobacco_consump_df[tobacco_consump_df['Measure'] == measure]

  # Pivot the filtered dataframe
  pivot = df_filtered.pivot_table(index='Year', columns='Country', values='Value', aggfunc='first')

  # Create a boolean dataframe indicating missing values
  missing_data_df = pivot.isna()

  # Visualize the missing data using a heatmap
  plt.figure(figsize=(20, 10))
  sns.heatmap(missing_data_df, cbar=False, linewidths=.1)
  plt.title(f'Missing Data in Tobacco Consumption - {measure}', fontsize=16)
  plt.xlabel('Country', fontsize=14)
  plt.ylabel('Year', fontsize=14)
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.subplots_adjust(bottom=0.3)  # Adjust if needed to prevent xlabel cutoff

  # Saving the heatmap
  heatmap_path = os.path.join(plots_directory, f'missing_data_heatmap_tobacco_{measure}.png')
  plt.savefig(heatmap_path)
  plt.close()

  print(f"Heatmap for {measure} saved to {heatmap_path}")
