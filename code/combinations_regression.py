import os
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pipeline import pipeline

# Constants and Configuration
DATA_PATH = '../data/merged_data.csv'
LINEAR_REGRESSION_DIR = '../data_visualization/linear_regression_fit'
MULTIVARIATE_REGRESSION_DIR = '../data_visualization/multivariate_regression'
# The amount of independent variables in the model
N_VARIABLES = 3
# Amount of standarddeviations in the outlier removal during fitting
NUM_STD_DEVS = 2


# Ensure directories exist
os.makedirs(LINEAR_REGRESSION_DIR, exist_ok=True)
os.makedirs(MULTIVARIATE_REGRESSION_DIR, exist_ok=True)

# Helper Functions
squared = lambda x: x**2
squared.__name__ = 'square'
reciprocal = lambda x: 1/x
reciprocal.__name__ = 'reciprocal'

# Transformations
transformations = [np.log, np.exp, np.sqrt, squared, reciprocal]

def load_data(path):
    """Load and return the data from a given path."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def plot_residuals_histogram(residuals, title, filename):
    """Plot and save the histogram of residuals."""
    plt.hist(residuals, bins=7)
    plt.title(title)
    plt.xlabel('Residual values')
    plt.ylabel('Frequencies')
    plt.savefig(filename)
    plt.close()

def plot_true_vs_predicted(y, y_predicted, title, filename):
    """Plot and save the true vs predicted values."""
    plt.scatter(y, y_predicted)
    plt.title(title)
    plt.ylabel('Predicted life expectancy')
    plt.xlabel('True life expectancy')
    plt.ylim(plt.xlim())
    plt.savefig(filename)
    plt.close()

def plot_residuals_vs_true(y, residuals, title, filename):
    """Plot and save the residuals vs true values."""
    plt.scatter(y, residuals)
    plt.title(title)
    plt.xlabel('Life expectancy')
    plt.ylabel('Residuals')
    plt.savefig(filename)
    plt.close()

def generate_combinations(elements, n):
    """Generate and return combinations of a given length from the provided list."""
    return list(combinations(elements, n))

# Main Analysis Function
def run_analysis(merged_df):
    """Run the regression analysis and plotting."""
    if merged_df is None:
        print("No data to analyze.")
        return

    # Prepare the data
    transformation_dict, prepared_data = pipeline(merged_df=merged_df)

    # Get unique drug variables and countries
    drug_vars = merged_df['Pharma_Sales_Variable'].unique().tolist()[1:]

    # Generate combinations of drugs for the models
    drug_combis = generate_combinations(drug_vars, N_VARIABLES)

    # Run the combinations of the models
    for df_current_gender in prepared_data:
        gender = df_current_gender['Life_Expectancy_Variable'].unique()[0]

        for pharma_sales_variables in drug_combis:

            X = np.zeros((len(df_current_gender['Country'].unique()), N_VARIABLES))

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
            initial_length = len(X)  # Store the initial length of the dataset
            total_outliers_removed = 0  # Initialize a counter for total outliers removed

            # Outlier removal loop
            while outliers_removed:
                # Fit the model
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                residuals = y - y_pred

                # Calculate standard deviation of residuals
                std_dev = np.std(residuals)

                # Identify non-outliers
                non_outlier_mask = np.abs(residuals) <= NUM_STD_DEVS * std_dev
                prev_len = len(X)
                X = X[non_outlier_mask.values]
                y = y[non_outlier_mask]

                # Update the total number of outliers removed
                outliers_removed_in_iteration = prev_len - len(X)
                total_outliers_removed += outliers_removed_in_iteration

                # Check if any outliers were removed in this iteration
                outliers_removed = len(X) < prev_len

                # Exit the loop if over two fifth of the original dataset has been removed in order ceep the model from overfitting too much
                if total_outliers_removed > initial_length / 2.5:
                    break


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
            # Refit the model with the final cleaned data
            model.fit(X_train, y_train)

            train_test_score = model.score(X_test, y_test)

            # Plot in 2D if only one explanatory variable
            if N_VARIABLES == 1:
                # Scatter plot of the random variable and the independent variable
                plt.scatter(X[:, 0], y)
                plt.xlabel(pharma_sales_variables)
                plt.ylabel(gender)
                plt.title(f'Fitted model for life expectancy - {gender} and pharma-sales - {drug}')

                # Plotting the fitted linear regression line
                x_values = np.linspace(min(X[:, 0]), max(X[:,0]), 10)
                plt.plot(x_values, model.intercept_ + model.coef_ * x_values, label="fitted line", color='orange')

                filename = f"{gender.replace(' ', '_') + pharma_sales_variables[0].replace(' ', '_')}"
                filepath = os.path.join(LINEAR_REGRESSION_DIR, filename)
                plt.legend()
                plt.savefig(filepath)
                plt.close()

            y_predicted = model.predict(X)
            residuals = y - y_predicted
            mse = np.mean(residuals**2)

            calculated_rsquared = 1 - np.sum((y - y_predicted)**2)/np.sum((y - np.mean(y))**2)
            adjusted_r = 1 - ((1 - calculated_rsquared) * (len(y) - 1)/(len(y) - 1 - N_VARIABLES))


            if mse < 0.1 and adjusted_r > 0.9 and train_test_score > 0.9:

                pharma_sales_variables_string = '_'.join([v.split('-')[0] if '-' in v else v.replace(' ', '_') for v in pharma_sales_variables])

                # Plot residuals histogram
                residuals_title = f'Residuals histogram of {pharma_sales_variables_string} {gender}'
                residuals_filename = os.path.join(MULTIVARIATE_REGRESSION_DIR, f"histogram_residuals_{gender.replace(' ', '_')}_{pharma_sales_variables_string}.png")
                plot_residuals_histogram(residuals, residuals_title, residuals_filename)

                # Plot true vs predicted values
                true_vs_predicted_title = f'The true vs predicted life expectancy of {pharma_sales_variables_string} {gender}'
                true_vs_predicted_filename = os.path.join(MULTIVARIATE_REGRESSION_DIR, f"true_predicted_{gender.replace(' ', '_')}_{pharma_sales_variables_string}.png")
                plot_true_vs_predicted(y, y_predicted, true_vs_predicted_title, true_vs_predicted_filename)


                # Plot residuals vs true values
                residuals_vs_true_title = f'The residuals with the true life expectancy {pharma_sales_variables_string} {gender}'
                residuals_vs_true_filename = os.path.join(MULTIVARIATE_REGRESSION_DIR, f"residuals_{gender.replace(' ', '_')}_{pharma_sales_variables_string}.png")
                plot_residuals_vs_true(y, residuals, residuals_vs_true_title, residuals_vs_true_filename)


                print(mse, "mean squared error")
                print(f"Results for Pharma Sales: {pharma_sales_variables_string}")
                print(f"Results for Life Expectancy: {gender}")
                print(model.intercept_)
                print(model.coef_)
                print("adjusted_r", adjusted_r)
                print("train_test_score", train_test_score)
                print('\n')

# Main Script Execution
if __name__ == "__main__":
    merged_df = load_data(DATA_PATH)
    run_analysis(merged_df)
