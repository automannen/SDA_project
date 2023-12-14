import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from tqdm import tqdm
from pipeline import pipeline
from scipy.stats import boxcox

squared = lambda x: x**2
squared.__name__ = 'square'
reciprocal = lambda x: 1/x
reciprocal.__name__ = 'reciprocal'

merged_df = pd.read_csv('merged_data.csv')

n_variables = 3
drug_vars = merged_df['Pharma_Sales_Variable'].unique().tolist()[1:]
drug_combis = list(combinations(drug_vars, n_variables))
print(len(drug_combis))

df = defaultdict(lambda: defaultdict(lambda: []))
for drug in tqdm(merged_df[["Pharma_Sales_Variable", "Pharma_Sales_Value", "Life_Expectancy_Variable", "Missingness_Indicator"]].iterrows()):
    df[drug[1]["Life_Expectancy_Variable"]][drug[1]["Pharma_Sales_Variable"]].append(drug[1]["Pharma_Sales_Value"])
    df[drug[1]["Life_Expectancy_Variable"]][drug[1]["Pharma_Sales_Variable"]+"missingness"].append(drug[1]["Missingness_Indicator"])

df = pd.DataFrame(df)
# print(df)
transformations = [np.log, np.exp, np.sqrt, squared, reciprocal, boxcox]
# Columns:
# Country,Pharma_Sales_Variable,Pharma_Sales_Value,Life_Expectancy_Variable,Life_Expectancy_Value,Missingness_Indicator

transformation_dict = pipeline()
print(transformation_dict)

# Run generalized linear models for each life expectancy variable
for life_exp_variable in tqdm(merged_df['Life_Expectancy_Variable'].unique()):

#   print(f"\nResults for Life Expectancy: {life_exp_variable}")
  current_life_exp_df = merged_df[merged_df['Life_Expectancy_Variable'] == life_exp_variable]

  for pharma_sales_variables in drug_combis:
    # print(f"Results for Pharma Sales: {pharma_sales_variables}")
    X = np.zeros((1650, n_variables*2))

    # filling the X matrix
    for i, drug in enumerate(pharma_sales_variables):
        X[:, i] = transformations[transformation_dict[life_exp_variable][drug][2]](np.array(df[life_exp_variable][drug]) + 1e-10)
        X[:, i + n_variables] = np.array(df[life_exp_variable][drug+"missingness"])

    y = current_life_exp_df[(current_life_exp_df["Pharma_Sales_Variable"] == list(pharma_sales_variables)[0])]
    y = y["Life_Expectancy_Value"]

    model = LinearRegression()
    model.fit(X, y)

    score = model.score(X, y)
    # threshold for the score
    if score > 0.5:
        print(f"Results for Pharma Sales: {pharma_sales_variables}")
        print(f"\nResults for Life Expectancy: {life_exp_variable}")
        print(model.intercept_)
        print(model.coef_)
        print("LETSGO", score)

