import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy.stats import norm
import statistics
import numpy as np
from scipy.stats import poisson

# TODO: plot te gebuiken in de presentatie
# plot om te kijken of life expectancy normaal verdeeld is

life_expectancy_df = pd.read_csv('../data/life_expectancy.csv')
filtered_life_expectancy_df = life_expectancy_df[
  (life_expectancy_df['Variable'].isin(['Females at age 40', 'Males at age 40'])) &
  (life_expectancy_df['Measure'] == 'Years') &
  (life_expectancy_df['Year'] == 2014)
]

selected_columns = filtered_life_expectancy_df[['Variable', 'Value']]

gender_life_expectancy = defaultdict(lambda: [])

for row in selected_columns.iterrows():
    gender_life_expectancy[row[1]["Variable"]].append(row[1]["Value"]*row[1]["Value"])

for gender in ['Females at age 40', 'Males at age 40']:
    data = gender_life_expectancy[gender]
    mean = np.mean(data)
    sd = np.std(data)
    x = np.linspace(min(data), max(data), 100)

    plt.plot(x, norm.pdf(x, mean, sd))
    plt.hist(gender_life_expectancy[gender], density=True, bins=20)
    plt.title(gender)
    plt.xlabel("life expectancy at age 40")
    plt.ylabel("bins count")
    plt.show()