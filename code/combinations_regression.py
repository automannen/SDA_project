import pandas as pd
import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
from itertools import combinations
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from tqdm import tqdm

merged_df = pd.read_csv('merged_data.csv')

n_variables = 5
drug_vars = merged_df['Pharma_Sales_Variable'].unique().tolist()
drug_combis = list(combinations(drug_vars, n_variables))

df = defaultdict(lambda: defaultdict(lambda: []))
for drug in tqdm(merged_df[["Pharma_Sales_Variable", "Pharma_Sales_Value", "Life_Expectancy_Variable"]].iterrows()):
    df[drug[1]["Life_Expectancy_Variable"]][drug[1]["Pharma_Sales_Variable"]].append(drug[1]["Pharma_Sales_Value"])

df = pd.DataFrame(df)
# print(df)

# Columns:
# Country,Pharma_Sales_Variable,Pharma_Sales_Value,Life_Expectancy_Variable,Life_Expectancy_Value,Missingness_Indicator


# Run generalized linear models for each life expectancy variable
for life_exp_variable in merged_df['Life_Expectancy_Variable'].unique():

#   print(f"\nResults for Life Expectancy: {life_exp_variable}")
  current_life_exp_df = merged_df[merged_df['Life_Expectancy_Variable'] == life_exp_variable]

  for pharma_sales_variables in drug_combis:
    # print(f"Results for Pharma Sales: {pharma_sales_variables}")
    X = np.zeros((1650, n_variables))

    # Filter the DataFrame to include only the current life expectancy variable and pharma sales variable
    for i, drug in enumerate(pharma_sales_variables):
        X[:, i] = np.array(df[life_exp_variable][drug])
    # print(current_life_exp_df)
    # print(list(pharma_sales_variables))

    y = current_life_exp_df[(current_life_exp_df["Pharma_Sales_Variable"] == list(pharma_sales_variables)[0])]
    y = y["Life_Expectancy_Value"]
    # print(np.shape(current_life_exp_df[["Life_Expectancy_Value"]]))

    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    if score > 0.5:
        print(f"Results for Pharma Sales: {pharma_sales_variables}")
        print(f"\nResults for Life Expectancy: {life_exp_variable}")
        print("LETSGO", score)

