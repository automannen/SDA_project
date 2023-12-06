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
plots_directory = 'food_heatmaps'
os.makedirs(plots_directory, exist_ok=True)

# Read the food consumption data
food_df = pd.read_csv(os.path.join(base_dir, 'food.csv'))

# Filter for specified countries and years
food_df = food_df[food_df["Country"].isin(countries) & food_df['Year'].isin(years)]

# Get unique variables and measures
variables = food_df['Variable'].unique()
measures = food_df['Measure'].unique()

# Create a heatmap for each variable and measure combination
for variable in variables:
  for measure in measures:
    # Filter the dataframe for the current variable and measure
    df_filtered = food_df[(food_df['Variable'] == variable) & (food_df['Measure'] == measure)]

    # If there's no data after filtering, skip this variable and measure
    if df_filtered.empty:
      print(f"No data for {variable} - {measure}. Skipping heatmap generation.")
      continue

    # Pivot the filtered dataframe
    pivot = df_filtered.pivot_table(index='Year', columns='Country', values='Value', aggfunc='first')

    # Create a boolean dataframe indicating missing values
    missing_data_df = pivot.isna()

    # Only attempt to create a heatmap if there are non-missing values
    if not missing_data_df.all().all():  # Checks if all values are True (missing)
      # Visualize the missing data using a heatmap
      plt.figure(figsize=(20, 10))
      sns.heatmap(missing_data_df, cbar=False, linewidths=.1)
      plt.title(f'Missing Data in Food Consumption - {variable} - {measure}', fontsize=16)
      plt.xlabel('Country', fontsize=14)
      plt.ylabel('Year', fontsize=14)
      plt.xticks(rotation=90)
      plt.tight_layout()
      plt.subplots_adjust(bottom=0.3)  # Adjust if needed to prevent xlabel cutoff

      # Saving the heatmap
      heatmap_path = os.path.join(plots_directory, f'missing_data_heatmap_food_{variable}_{measure}.png')
      plt.savefig(heatmap_path)
      plt.close()

      print(f"Heatmap for {variable} - {measure} saved to {heatmap_path}")
    else:
      print(f"All data missing for {variable} - {measure}. Skipping heatmap generation.")
