import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from collections import defaultdict
import os
from sklearn.linear_model import LinearRegression

plots_directory = '../data_visualization/transformed_data'
os.makedirs(plots_directory, exist_ok=True)
pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')
merged_df = pd.read_csv('../data/merged_data.csv')
drugs = pharma_sales_df['Variable'].unique()[1:] # skipping the total sales variable
squared = lambda x: x**2
squared.__name__ = 'square'
reciprocal = lambda x: 1/x
reciprocal.__name__ = 'reciprocal'

transformations = [np.log, np.exp, np.sqrt, squared, reciprocal]

def plot_transformed_data(x, y, best_transformation, drug, gender, model):

    if best_transformation == None:
        plt.scatter(x, y)
        plt.title(f'{drug} not transformed')
    else:
        plt.title(f'{drug} transformed with {best_transformation.__name__}')
        plt.scatter(best_transformation(x), y)

    filename = f"{'old_pipeline_'+ drug.replace(' ', '_')}_{gender.replace(' ', '_')}"
    filepath = os.path.join(plots_directory, filename)

    plt.xlabel(drug)
    plt.ylabel(gender)
    plt.savefig(filepath)
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

            info_drug_x = merged_df[(merged_df['Life_Expectancy_Variable'] == gender) & (merged_df['Pharma_Sales_Variable'] == drug)][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
            best_transformation = None
            x = np.array(info_drug_x['Pharma_Sales_Value'])
            y = np.array(info_drug_x['Life_Expectancy_Value'])

            # Adding a small constant to circumvent the zero values in the data
            min_val = min(x)
            x = x - min_val + 1e-10

            # calculate the pearson correlation of the data without transformation
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

            model = LinearRegression()

            if best_transformation == None:
                model.fit(x.reshape(-1, 1), y)
                transformation_dict[gender][drug] = (best_transformation, pearson, transformation_idx)
            else:
                model.fit(best_transformation(x).reshape(-1, 1), y)

                # Saving the best transformation for each drug
                transformation_dict[gender][drug] = (best_transformation.__name__, pearson, transformation_idx)

            plot_transformed_data(x, y, best_transformation, drug, gender, model)

    return transformation_dict

print(pipeline())

