import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from tqdm import tqdm
from new_pipeline import pipeline
import os
import matplotlib.pyplot as plt

plots_directory = '../data_visualization/linear_regression_fit'
squared = lambda x: x**2
squared.__name__ = 'square'
reciprocal = lambda x: 1/x
reciprocal.__name__ = 'reciprocal'

extended = False

if extended:
    merged_df = pd.read_csv('merged_data_extended.csv')
    n_extended = 50
else:
    merged_df = pd.read_csv('merged_data.csv')
    n_extended = 1

# how many independent variables in the linear regression
n_variables = 1

num_std_devs = 2
drug_vars = merged_df['Pharma_Sales_Variable'].unique().tolist()[1:]

countries = merged_df['Country'].unique().tolist()
drug_combis = list(combinations(drug_vars, n_variables))

transformations = [np.log, np.exp, np.sqrt, squared, reciprocal]
transformations_inverse = [np.exp, np.log, squared, np.sqrt, reciprocal]
# Columns:
# Country,Pharma_Sales_Variable,Pharma_Sales_Value,Life_Expectancy_Variable,Life_Expectancy_Value,Missingness_Indicator

print("start pipeline")
transformation_dict, prepared_data = pipeline(merged_df=merged_df, n_extended=n_extended)
print('end pipeline')

# Run generalized linear models for each life expectancy variable
for df_current_gender in prepared_data:

#   print(f"\nResults for Life Expectancy: {life_exp_variable}")
  gender = df_current_gender['Life_Expectancy_Variable'].unique()[0]


  for pharma_sales_variables in drug_combis:
    # print(f"Results for Pharma Sales: {pharma_sales_variables}")

    X = np.zeros((len(df_current_gender['Country'].unique())*n_extended, n_variables))

    # filling the X matrix
    for i, drug in enumerate(pharma_sales_variables):
        if transformation_dict[gender][drug][2] == None:
           X[:, i] = df_current_gender[df_current_gender['Pharma_Sales_Variable'] == drug]['Pharma_Sales_Value']
        else:
            transformation_idx = transformation_dict[gender][drug][2]
            func = transformations[transformation_idx]
            # adding a small constant for the log transformation
            small_const = 1e-10
            X[:, i] = func(df_current_gender[df_current_gender['Pharma_Sales_Variable'] == drug]['Pharma_Sales_Value'] + small_const)

    y = df_current_gender[(df_current_gender["Pharma_Sales_Variable"] == list(pharma_sales_variables)[0])]
    y = y["Life_Expectancy_Value"]

    if n_variables == 1:
        plt.scatter(X[:, 0], y)
        plt.xlabel(pharma_sales_variables)
        plt.ylabel(gender)
        plt.title('fitted line in plot')

    outliers_removed = True

    while outliers_removed:
        # Fit the model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Calculate standard deviation of residuals
        std_dev = np.std(residuals)

        # Identify non-outliers
        non_outlier_mask = np.abs(residuals) <= num_std_devs * std_dev
        prev_len = len(X)
        X = X[non_outlier_mask.values]
        y = y[non_outlier_mask]

        # Check if any outliers were removed in this iteration
        outliers_removed = len(X) < prev_len

    # Refit the model with the final cleaned data
    model.fit(X, y)

    score = model.score(X, y)
    # threshold for the score

    if n_variables == 1:
        x_values = np.linspace(min(X[:, 0]), max(X[:,0]), 10)
        plt.plot(x_values, model.intercept_ + model.coef_ * x_values, label="fitted line", color='orange')
        filename = f"{gender.replace(' ', '_') + pharma_sales_variables[0].replace(' ', '_')}"
        filepath = os.path.join(plots_directory, filename)
        plt.savefig(filepath)
        plt.legend()
        plt.close()


    coeffs = []

    if min(model.coef_) >= 0.1:

        for i, drug in enumerate(pharma_sales_variables):
            if transformation_dict[gender][drug][2] == None:
                coeffs.append(model.coef_[i])
            else:

                transformation_idx = transformation_dict[gender][drug][2]
                func = transformations_inverse[transformation_idx]
                # adding a small constant for the log transformation
                small_const = 1e-10
                coeffs.append(func(model.coef_[i] + small_const))
    if len(coeffs) != 0 and min(coeffs) >= 0.5:
        print(f"Results for Pharma Sales: {pharma_sales_variables}")
        print(f"Results for Life Expectancy: {gender}")
        print(model.intercept_)
        print(model.coef_)
        print(coeffs, "Transformed")
        print("LETSGO", score)
        print('\n')



