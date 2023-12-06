import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# List of countries and years of interest
countries = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Czechia',
             'Estonia', 'Finland', 'Germany', 'Greece', 'Hungary', 'Iceland',
             'Ireland', 'Italy', 'Korea', 'Latvia', 'Luxembourg', 'Netherlands',
             'New Zealand', 'Norway', 'Portugal', 'Slovak Republic', 'Slovenia',
             'Spain', 'Sweden', 'Türkiye']
years = [2011, 2012, 2013, 2014, 2016]

# Base directory for CSV files
base_dir = '../data/'  # Update this to the path where your CSV files are stored

# Directory for saving plots
plots_directory = 'alcohol_tobacco_heatmaps'
os.makedirs(plots_directory, exist_ok=True)

# Read and filter both dataframes
tobacco_consump_df = pd.read_csv(os.path.join(base_dir, 'tobacco_consump.csv'))
alcohol_consump_df = pd.read_csv(os.path.join(base_dir, 'alcohol_consump.csv'))

# Filter for specified countries and years
tobacco_consump_df = tobacco_consump_df[tobacco_consump_df["Country"].isin(countries) & tobacco_consump_df['Year'].isin(years)]
alcohol_consump_df = alcohol_consump_df[alcohol_consump_df["Country"].isin(countries) & alcohol_consump_df['Year'].isin(years)]

# # Pivot the dataframes

alcohol_pivot = alcohol_consump_df.pivot_table(index='Year', columns='Country', values='Value', aggfunc='first')

# # Rename the columns for clarity

alcohol_pivot.columns = [f'{country} - Alcohol' for country in alcohol_pivot.columns]

# Pivot the tobacco dataframe including the 'Measure' column
tobacco_pivot = tobacco_consump_df.pivot_table(index='Year', columns=['Country', 'Measure'], values='Value', aggfunc='first')

# Flatten the MultiIndex in columns after pivoting
tobacco_pivot.columns = [' - '.join(col).strip() for col in tobacco_pivot.columns.values]

# Rename the columns for clarity, incorporating the Measure
tobacco_pivot.columns = [f'{measure} - Tobacco' for measure in tobacco_pivot.columns]

# Combine the pivoted dataframes
combined_pivot = pd.concat([tobacco_pivot, alcohol_pivot], axis=1)

# # Create a boolean dataframe indicating missing values
missing_data_df = combined_pivot.isna()


# country_names = [name.split(" -")[0] for name in missing_data_df.columns]
# # Now get the unique country names
# unique_countries = sorted(set(country_names))
# print(unique_countries)
# print(set(countries) - set(unique_countries))

# Visualize the missing data using a heatmap
plt.figure(figsize=(20, 10))  # Adjust the size as needed
sns.heatmap(missing_data_df, cbar=False, linewidths=.1)
plt.title('Missing Data in Tobacco and Alcohol Consumption by Year and Country', fontsize=16)
plt.xlabel('Country - Variable', fontsize=14)
plt.ylabel('Year', fontsize=14)
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Adjust if needed to prevent xlabel cutoff

heatmap_path = os.path.join(plots_directory, 'missing_data_heatmap.png')
plt.savefig(heatmap_path)
plt.close()

print(f"Heatmap saved to {heatmap_path}")
