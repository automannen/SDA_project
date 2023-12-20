import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict
import os
from outlier_detection import detect_and_remove_outliers

plots_directory = '../data_visualization/transformed_data'
pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')
merged_df = pd.read_csv('merged_data.csv')
drugs = pharma_sales_df['Variable'].unique()[1:] # skipping the total sales variable
squared = lambda x: x**2
squared.__name__ = 'square'
reciprocal = lambda x: 1/x
reciprocal.__name__ = 'reciprocal'

transformations = [np.log, np.exp, np.sqrt, squared, reciprocal]

def plot_transformed_data(x, y, best_transformation, drug, gender):

    if best_transformation == None:
        filename = f"{drug}_{gender}"
        plt.scatter(x, y)
        plt.title('no transformation')
    else:
        plt.title(best_transformation.__name__)
        plt.scatter(best_transformation(x), y)
        filename = f"{drug.replace(' ', '_')}_{gender.replace(' ', '_')}"

    filepath = os.path.join(plots_directory, filename)
    plt.xlabel(drug + 'transformed with some function')
    plt.ylabel(gender)
    plt.savefig(filepath)
    plt.close()


def pipeline(merged_df, n_extended=1):
    """
    This function takes different transformations on the sets of independent variables to find which is the best fit to the dependent variable.
    To find the best transformation, the pearson correlation coeffcient is used. This coefficient indicates the strenght and direction of the linear
    relationship with the life expectancy, r = 1 or -1 indicates a perfect linear relationship, r = 0 indicates no linear relationship.
    The function returns a dictionary with the best transformation for each drug.
    """
    transformation_dict = defaultdict(lambda: {})

    prepared_data = detect_and_remove_outliers(n_components = 2, n_neighbors = 2, f_std = 0, merged_df=merged_df, n_extended=n_extended)

    for df_current_gender in prepared_data:

        for drug in drugs:

            info_drug_x = df_current_gender[df_current_gender['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
            best_transformation = None

            x = np.array(info_drug_x['Pharma_Sales_Value'])
            y = np.array(info_drug_x['Life_Expectancy_Value'])

            # Adding a small constant to circumvent the zero values in the data
            min_val = min(x)
            x = x - min_val + 1e-10

            pearson, _ = pearsonr(x, y)
            best_transformation = None
            transformation_idx = None

            for idx, transformation in enumerate(transformations):
                transformed_data = transformation(x)
                new_pearson, _ = pearsonr(transformed_data, y)
                if abs(new_pearson) > abs(pearson): 
                    best_transformation = transformation
                    pearson = new_pearson
                    transformation_idx = idx

            if best_transformation == None:
                transformation_dict[df_current_gender['Life_Expectancy_Variable'].unique()[0]][drug] = (best_transformation, pearson, transformation_idx)
            else:
                transformation_dict[df_current_gender['Life_Expectancy_Variable'].unique()[0]][drug] = (best_transformation.__name__, pearson, transformation_idx)

    return transformation_dict, prepared_data

