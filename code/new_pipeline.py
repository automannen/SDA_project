import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, boxcox
from collections import defaultdict
import os
from sklearn.linear_model import LinearRegression
from outlier_detection_olivier import detect_and_remove_outliers

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

    # for i in range(19):
    #     max_value = max(x)
    #     idx_max = np.where(x == max_value)
    #     x = np.delete(x, idx_max)
    #     y = np.delete(y, idx_max)

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


def pipeline():
    """
    This function takes different transformations on the sets of independent variables to find which is the best fit to the dependent variable.
    To find the best transformation, the pearson correlation coeffcient is used. This coefficient indicates the strenght and direction of the linear
    relationship with the life expectancy, r = 1 or -1 indicates a perfect linear relationship, r = 0 indicates no linear relationship.
    The function returns a dictionary with the best transformation for each drug.
    """
    transformation_dict = defaultdict(lambda: {})

    prepared_data = detect_and_remove_outliers(n_components = 2, n_neighbors = 3, f_std = 0)

    for i, gender in enumerate(merged_df['Life_Expectancy_Variable'].unique()):

        df_current_gender =  prepared_data[i]
        for drug in drugs:

            info_drug_x = df_current_gender[df_current_gender['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
            best_transformation = None
            # filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]

            x = np.array(info_drug_x['Pharma_Sales_Value'])
            y = np.array(info_drug_x['Life_Expectancy_Value'])

            # Adding a small constant to circumvent the zero values in the data
            min_val = min(x)
            x = x - min_val + 1e-10

            # Set to 0 so there is always an improvement
            pearson, p_value = pearsonr(x, y)
            best_transformation = None
            # pearson = new_pearson
            best_p_value = p_value
            transformation_idx = None
            # best_p_value = 0
            # transformation_idx = None


            for idx, transformation in enumerate(transformations):
                if transformation == boxcox and drug != 'Total pharmaceutical sales':
                    transformed_data, _ = boxcox(x)
                else:
                    transformed_data = transformation(x)
                    new_pearson, p_value = pearsonr(transformed_data, y)
                if abs(new_pearson) > abs(pearson): #maakt niet uit of nega of posi
                    best_transformation = transformation
                    pearson = new_pearson
                    best_p_value = p_value
                    transformation_idx = idx

            # vanilla_pearson, p_value = pearsonr(x, y)
            # if abs(vanilla_pearson) > abs(pearson):
            #     best_transformation = None
            #     pearson = new_pearson
            #     best_p_value = p_value
            #     transformation_idx = None
            model = LinearRegression()

            if best_transformation == None:
                model.fit(x.reshape(-1, 1), y)
                transformation_dict[gender][drug] = (best_transformation, pearson, transformation_idx)
                # print("R^2: ", model.score(x.reshape(-1, 1), y), "DE SCORE")
            else:

                model.fit(best_transformation(x).reshape(-1, 1), y)
                # print("R^2: ", model.score(best_transformation(x).reshape(-1, 1), y), "DE SCORE")

            # Saving the best transformation for each drug
                transformation_dict[gender][drug] = (best_transformation.__name__, pearson, transformation_idx)
            plot_transformed_data(x, y, best_transformation, drug, gender)
            # if best_p_value <= 0.05:
            #     print("REJECTED")
            #     print(drug)
            #     print(gender)
            # else:
            #     print("ACCEPTED")
            #     print(drug)
            #     print(gender)

    return transformation_dict

print(pipeline())

