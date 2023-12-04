import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the directory where plots will be saved
plots_directory = "life_expectancy_plots"
os.makedirs(plots_directory, exist_ok=True)  # Create the directory if it does not exist

# Read the CSV file
df = pd.read_csv('../data/life_expectancy.csv')

# Convert 'Year' to numeric
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Loop through each combination of country, variable, and measure
for (variable, measure, country), group_df in df.groupby(['Variable', 'Measure', 'Country']):

    # Skip any group that is not measured in "Years"
    if measure != "Years":
        continue

    # Sort the dataframe by year for consistent plotting
    group_df = group_df.sort_values('Year')

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(group_df['Year'], group_df['Value'], marker='o')
    plt.title(f'Life Expectancy in {country} - {variable}')
    plt.xlabel('Year')
    plt.ylabel('Life Expectancy')

    # Define the filename with a path to the plots directory
    filename = f"{country}_{variable.replace(' ', '_')}_{measure.replace(' ', '_')}.png"
    filepath = os.path.join(plots_directory, filename)

    # Save the plot to the specified directory
    plt.savefig(filepath)
    plt.close()  # Close the figure after saving to avoid display
