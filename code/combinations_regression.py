import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from tqdm import tqdm
from pipeline import pipeline
import os
import matplotlib.pyplot as plt

linear_regression_fit_plots_directory = '../data_visualization/linear_regression_fit'
os.makedirs(linear_regression_fit_plots_directory, exist_ok=True)

multivariate_regression_plots_directory = '../data_visualization/multivariate_regression'
os.makedirs(multivariate_regression_plots_directory, exist_ok=True)

squared = lambda x: x**2
squared.__name__ = 'square'
reciprocal = lambda x: 1/x
reciprocal.__name__ = 'reciprocal'

merged_df = pd.read_csv('../data/merged_data.csv')

# The amount of independent variables in the model
n_variables = 3
# Amount of standarddeviations in the outlier removal during fitting
num_std_devs = 2

drug_vars = merged_df['Pharma_Sales_Variable'].unique().tolist()[1:]
countries = merged_df['Country'].unique().tolist()

# Form the combinations for all the models
drug_combis = list(combinations(drug_vars, n_variables))

transformations = [np.log, np.exp, np.sqrt, squared, reciprocal]
# transformations_inverse = [np.exp, np.log, squared, np.sqrt, reciprocal]

# Columns:
# Country,Pharma_Sales_Variable,Pharma_Sales_Value,Life_Expectancy_Variable,Life_Expectancy_Value,Missingness_Indicator

transformation_dict, prepared_data = pipeline(merged_df=merged_df)

# Run the combinations of the models
for df_current_gender in prepared_data:
  gender = df_current_gender['Life_Expectancy_Variable'].unique()[0]

  for pharma_sales_variables in drug_combis:

    X = np.zeros((len(df_current_gender['Country'].unique()), n_variables))

    # Filling the X matrix
    for i, drug in enumerate(pharma_sales_variables):
        if transformation_dict[gender][drug][2] == None:

           X[:, i] = df_current_gender[df_current_gender['Pharma_Sales_Variable'] == drug]['Pharma_Sales_Value']
        else:

            transformation_idx = transformation_dict[gender][drug][2]
            func = transformations[transformation_idx]
            small_const = 1e-10
            X[:, i] = func(df_current_gender[df_current_gender['Pharma_Sales_Variable'] == drug]['Pharma_Sales_Value'] + small_const)

    y = df_current_gender[(df_current_gender["Pharma_Sales_Variable"] == list(pharma_sales_variables)[0])]
    y = y["Life_Expectancy_Value"]

    outliers_removed = True

    # Outlier removal
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

    # Plot in 2D if only one explanatory variable
    if n_variables == 1:
        # Scatter plot of the random variable and the independent variable
        plt.scatter(X[:, 0], y)
        plt.xlabel(pharma_sales_variables)
        plt.ylabel(gender)
        plt.title(f'Fitted model for life expectancy - {gender} and pharma-sales - {drug}')

        # Plotting the fitted linear regression line
        x_values = np.linspace(min(X[:, 0]), max(X[:,0]), 10)
        plt.plot(x_values, model.intercept_ + model.coef_ * x_values, label="fitted line", color='orange')

        filename = f"{gender.replace(' ', '_') + pharma_sales_variables[0].replace(' ', '_')}"
        filepath = os.path.join(linear_regression_fit_plots_directory, filename)
        plt.legend()
        plt.savefig(filepath)
        plt.close()

    y_predicted = model.predict(X)
    residuals = y - y_predicted
    mse = np.mean(residuals**2)

    calculated_rsquared = 1 - np.sum((y - y_predicted)**2)/np.sum((y - np.mean(y))**2)
    adjusted_r = 1 - ((1 - calculated_rsquared) * (len(y) - 1)/(len(y) - 1 - n_variables))
    print(mse, adjusted_r)
    if mse < 0.1 and adjusted_r > 0.9:

        pharma_sales_variables_string = '_'.join([v.split('_')[0] if '_' in v else v.replace(' ', '_') for v in pharma_sales_variables])

        plt.hist(residuals, bins=7)
        plt.title(f'Residuals histogram plot of {pharma_sales_variables_string} {gender}')
        plt.xlabel('residual values')
        plt.ylabel('frequencies')

        filename = f"histogram_residuals_{gender.replace(' ', '_')}_{pharma_sales_variables_string}"
        filepath = os.path.join(multivariate_regression_plots_directory, filename)
        plt.savefig(filepath)
        plt.close()


        plt.scatter(y, y_predicted)
        plt.title(f'The true life expectancy vs the predicted life expectancy of {gender}')
        plt.ylabel(f'predicted life expectancy')
        plt.xlabel(f'true life expectancy')
        plt.ylim(plt.xlim())

        filename = f"true_predicted_{gender.replace(' ', '_')}_{pharma_sales_variables_string}"
        filepath = os.path.join(multivariate_regression_plots_directory, filename)
        plt.savefig(filepath)
        plt.close()


        plt.scatter(y, residuals)
        plt.title(f'The residuals plotted with the true life expectancy {gender}')
        plt.xlabel('life expectancy')
        plt.ylabel('Residuals')

        filename = f"residuals_{gender.replace(' ', '_')}_{pharma_sales_variables_string}"
        filepath = os.path.join(multivariate_regression_plots_directory, filename)
        plt.savefig(filepath)
        plt.close()

        print(mse, "mean squared error")
        print(f"Results for Pharma Sales: {pharma_sales_variables_string}")
        print(f"Results for Life Expectancy: {gender}")
        print(model.intercept_)
        print(model.coef_)
        print("LETSGO", score)
        print('\n')



