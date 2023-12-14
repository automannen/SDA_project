import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, boxcox
from collections import defaultdict
import os

plots_directory = '../data_visualization/transformed_data'
pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')
merged_df = pd.read_csv('merged_data.csv')
drugs = pharma_sales_df['Variable'].unique()[1:] # skipping the total sales variable
squared = lambda x: x**2
squared.__name__ = 'square'
reciprocal = lambda x: 1/x
reciprocal.__name__ = 'reciprocal'

transformations = [np.log, np.exp, np.sqrt, squared, reciprocal, boxcox]

def plot_transformed_data(x, y, best_transformation, drug, gender):
    plt.scatter(best_transformation(x), y)
    filename = f"{drug}_{gender}_{best_transformation.__name__}"
    filepath = os.path.join(plots_directory, filename)
    plt.savefig(filepath)
    plt.title(best_transformation.__name__)
    plt.xlabel(drug)
    plt.ylabel(gender)
    plt.close()



def pipeline():
    """
    This function takes different transformations on the sets of independent variables to find which is the best fit to the dependent variable.
    To find the best transformation, the pearson correlation coeffcient is used. This coefficient indicates the strenght and direction of the linear
    relationship with the life expectancy, r = 1 or -1 indicates a perfect linear relationship, r = 0 indicates no linear relationship.
    The function returns a dictionary with the best transformation for each drug.
    """
    transformation_dict = defaultdict(lambda: {})

    for gender in merged_df['Life_Expectancy_Variable'].unique():

        for drug in drugs:

            info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
            best_transformation = None
            filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]

            x = np.array(filter_gender['Pharma_Sales_Value'])
            y = np.array(filter_gender['Life_Expectancy_Value'])

            # Adding a small constant to circumvent the zero values in the data
            x = x + 1e-10

            # Set to 0 so there is always an improvement
            pearson = 0
            best_p_value = 0

            for transformation in transformations:
                if transformation == boxcox:
                    transformed_data, _ = boxcox(x)
                else:
                    transformed_data = transformation(x)
                    new_pearson, p_value = pearsonr(transformed_data, y)
                if abs(new_pearson) > abs(pearson): #maakt niet uit of nega of posi
                    best_transformation = transformation
                    pearson = new_pearson
                    best_p_value = p_value


            # Saving the best transformation for each drug
            transformation_dict[gender][drug] = (best_transformation.__name__, pearson, best_p_value)
            plot_transformed_data(x, y, best_transformation, drug, gender)
            if best_p_value <= 0.05:
                print("REJECTED")
                print(drug)
                print(gender)
            else:
                print("ACCEPTED")
                print(drug)
                print(gender)

    return transformation_dict

print(pipeline())

