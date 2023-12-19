import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


merged_df = pd.read_csv('merged_data.csv')
drugs = merged_df['Pharma_Sales_Variable'].unique()[1:]# skipping the total sales variable
countries = merged_df['Country'].unique()


# Function to identify single-item clusters using Nearest Neighbors
def identify_single_item_clusters_nn(X, n_components, n_neighbors, f_std):
    # Applying PCA
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)

    # Initialize NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X_pca)

    # Calculate the distances and indices of neighbors
    distances, indices = neigh.kneighbors(X_pca)

    # Determine the average distance to neighbors for each point
    avg_distances = np.mean(distances, axis=1)

    # Identifying potential single-item clusters
    # You can adjust the threshold based on your specific needs
    threshold = np.mean(avg_distances) + f_std * np.std(avg_distances)
    single_item_indices = np.where(avg_distances > threshold)[0]

    return single_item_indices, X_pca


def detect_and_remove_outliers(n_components = 2, n_neighbors = 5, f_std = 0.15):
    single_item_clusters_info_nn = {}
    remainers = {}

    for gender in merged_df['Life_Expectancy_Variable'].unique():
        X = np.zeros((len(countries), len(drugs)))

        for i, drug in enumerate(drugs):
            # print(i, drug)
            info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
            filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]

            x = np.array(filter_gender['Pharma_Sales_Value'])
            X[:, i] = x


        # print(gender)

        # Identifying single-item clusters using nearest neighbors after PCA
        single_item_indices_nn, X_pca = identify_single_item_clusters_nn(X, n_components, n_neighbors, f_std)
        # print(single_item_indices_nn)

        # TODO: uncomment to plot
        # scatterplot all dots without single_item_indices_nn
        # for i in range(len(X_pca)):
        #     if i not in single_item_indices_nn:
        #         plt.scatter(X_pca[i, 0], X_pca[i, 1], label=i, color='blue')
        #     else:
        #         plt.scatter(X_pca[i, 0], X_pca[i, 1], label=i, color='red')
        # plt.show()

        single_item_clusters_info_nn[gender] = single_item_indices_nn

        non_outlier_countries = np.delete(countries, single_item_indices_nn)

        # for i, county in enumerate(countries):
        #     if gender in single_item_clusters_info_nn and i in single_item_clusters_info_nn[gender]:
        #         print(county, "is an outlier")
        # print("non_outlier_countries", non_outlier_countries)
        # print(non_outlier_countries)
        remainers[gender] = non_outlier_countries

    female_df = merged_df[(merged_df["Life_Expectancy_Variable"] == "Females at age 40") & (merged_df['Country'].isin(remainers["Females at age 40"]))]
    male_df = merged_df[(merged_df["Life_Expectancy_Variable"] == "Males at age 40") & (merged_df['Country'].isin(remainers["Males at age 40"]))]

    return (male_df, female_df)




# detect_and_remove_outliers()