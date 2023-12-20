# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
# import os


# merged_df = pd.read_csv('merged_data.csv')
# drugs = merged_df['Pharma_Sales_Variable'].unique()[1:]  # skipping the total sales variable
# countries = merged_df['Country'].unique()

# # Directory for saving plots
# plots_directory = '../data_visualization/clustering'
# os.makedirs(plots_directory, exist_ok=True)


# def identify_single_item_clusters_nn(X, n_components, n_neighbors, f_std):
#     pca = PCA(n_components)
#     X_pca = pca.fit_transform(X)

#     # Initialize NearestNeighbors
#     neigh = NearestNeighbors(n_neighbors=n_neighbors)
#     neigh.fit(X_pca)

#     # Calculate the distances of neighbors
#     distances, _ = neigh.kneighbors(X_pca)

#     # Determine the average distance to neighbors for each point
#     avg_distances = np.mean(distances, axis=1)

#     # Identifying distant points
#     threshold = np.mean(avg_distances) + f_std * np.std(avg_distances)
#     single_item_indices = np.where(avg_distances > threshold)[0]

#     return single_item_indices, X_pca


# def detect_and_remove_outliers(n_components = 2, n_neighbors = 5, f_std = 0.15, merged_df=merged_df, n_extended=1):
#     remainers = {}

#     for gender in merged_df['Life_Expectancy_Variable'].unique():
#         X = np.zeros((len(countries) * n_extended, len(drugs)))
#         for i, drug in enumerate(drugs):
#             info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
#             filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]

#             x = np.array(filter_gender['Pharma_Sales_Value'])
#             X[:, i] = x


#         single_item_indices_nn, X_pca = identify_single_item_clusters_nn(X, n_components, n_neighbors, f_std)


#         # scatterplot all dots without single_item_indices_nn
#         InliersLabel = True
#         OutlierLabel = True
#         for i in range(len(X_pca)):
#             if i not in single_item_indices_nn:
#                 plt.scatter(X_pca[i, 0], X_pca[i, 1], label=("","Inliers")[InliersLabel], color='blue')
#                 if InliersLabel: InliersLabel = False
#             else:
#                 plt.scatter(X_pca[i, 0], X_pca[i, 1], label=("","Outliers")[OutlierLabel], color='red')
#                 if OutlierLabel: OutlierLabel = False

#        # Some combination of drug-related features that contributed most to the variability in the data
#         plt.xlabel('First principal component values')
#         # Another composite measure/combination of features, capturing remaining variability
#         plt.ylabel('Second principal component values')

#         plt.legend()
#         plt.title('Outliers KNN ' + gender)

#         plt_name = 'KNN_' + str(gender.replace(' ', '_')) + '.png'
#         heatmap_path = os.path.join(plots_directory, plt_name)
#         plt.savefig(heatmap_path)
#         plt.close()

#         non_outlier_countries = np.delete(countries, single_item_indices_nn)

#         remainers[gender] = non_outlier_countries

#     female_df = merged_df[(merged_df["Life_Expectancy_Variable"] == "Females at age 40") & (merged_df['Country'].isin(remainers["Females at age 40"]))]
#     male_df = merged_df[(merged_df["Life_Expectancy_Variable"] == "Males at age 40") & (merged_df['Country'].isin(remainers["Males at age 40"]))]

#     return (male_df, female_df)


# detect_and_remove_outliers()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import os

# Read the merged data from the CSV file
merged_df = pd.read_csv('merged_data.csv')

# Extract unique pharmaceutical sales variables and countries
drugs = merged_df['Pharma_Sales_Variable'].unique()[1:]  # Skipping the total sales variable
countries = merged_df['Country'].unique()

# Directory for saving plots, created if it doesn't exist
plots_directory = '../data_visualization/clustering'
os.makedirs(plots_directory, exist_ok=True)

def identify_single_item_clusters_nn(X, n_components, n_neighbors, f_std):
    """Identifies single item clusters using PCA and Nearest Neighbors."""
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_pca)

    distances, _ = neigh.kneighbors(X_pca)
    avg_distances = np.mean(distances, axis=1)

    # Identifying distant points
    threshold = np.mean(avg_distances) + f_std * np.std(avg_distances)
    single_item_indices = np.where(avg_distances > threshold)[0]

    return single_item_indices, X_pca

def detect_and_remove_outliers(n_components=2, n_neighbors=5, f_std=0.15, merged_df=merged_df, n_extended=1):
    """Detects and removes outliers from the dataset."""
    remainers = {}

    for gender in merged_df['Life_Expectancy_Variable'].unique():
        X = np.zeros((len(countries) * n_extended, len(drugs)))

        for i, drug in enumerate(drugs):
            info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug]
            filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]
            X[:, i] = filter_gender['Pharma_Sales_Value'].values

        single_item_indices_nn, X_pca = identify_single_item_clusters_nn(X, n_components, n_neighbors, f_std)

        plt.figure()
        InliersLabel = True
        OutlierLabel = True
        for i in range(len(X_pca)):
            color, label = ('blue', ("", "Inliers")[InliersLabel]) if i not in single_item_indices_nn else ('red', ("", "Outliers")[OutlierLabel])
            plt.scatter(X_pca[i, 0], X_pca[i, 1], label=label, color=color)
            if i not in single_item_indices_nn:
                InliersLabel = False
            else:
                OutlierLabel = False

        plt.xlabel('First principal component values')
        plt.ylabel('Second principal component values')
        plt.legend()
        plt.title(f'Outliers KNN {gender}')
        plt_name = f'KNN_{gender.replace(" ", "_")}.png'
        heatmap_path = os.path.join(plots_directory, plt_name)
        plt.savefig(heatmap_path)
        plt.close()

        non_outlier_countries = np.delete(countries, single_item_indices_nn)
        remainers[gender] = non_outlier_countries

    # Create dataframes for males and females without outliers
    female_df = merged_df[(merged_df["Life_Expectancy_Variable"] == "Females at age 40") &
                          (merged_df['Country'].isin(remainers["Females at age 40"]))]
    male_df = merged_df[(merged_df["Life_Expectancy_Variable"] == "Males at age 40") &
                        (merged_df['Country'].isin(remainers["Males at age 40"]))]

    return male_df, female_df

detect_and_remove_outliers()
