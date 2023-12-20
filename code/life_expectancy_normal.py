# Plotting the life expectancy in a histogram to check if it is normally distributed

import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
import numpy as np
import os

n_datapoints = 100

plots_directory = '../data_visualization/life_expectancy'
os.makedirs(plots_directory, exist_ok=True)

life_expectancy_df = pd.read_csv('../data/life_expectancy.csv')
filtered_life_expectancy_df = life_expectancy_df[
  (life_expectancy_df['Variable'].isin(['Females at age 40', 'Males at age 40'])) &
  (life_expectancy_df['Measure'] == 'Years') &
  (life_expectancy_df['Year'] == 2014)
]

selected_columns = filtered_life_expectancy_df[['Variable', 'Value']]
selected_columns = pd.DataFrame(selected_columns)

for gender in selected_columns['Variable'].unique():

    data = selected_columns[selected_columns['Variable'] == gender][['Value']]
    mean = np.mean(data)
    sd = np.std(data)

    x = np.linspace(min(np.array(data)), max(np.array(data)), n_datapoints)

    plt.plot(x, norm.pdf(x, mean, sd))
    plt.hist(data, density=True, bins=20)
    plt.title(f"Histogram of {gender}")
    plt.xlabel(gender)
    plt.ylabel("Normalized frequency")
    filename = f"{gender}"
    filepath = os.path.join(plots_directory, filename)
    plt.savefig(filepath)
    plt.close()
