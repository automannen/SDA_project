import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


merged_df = pd.read_csv('merged_data.csv')
drugs = merged_df['Pharma_Sales_Variable'].unique()[1:]# skipping the total sales variable
countries = merged_df['Country'].unique()


# Parameters
n_components = 2  # Number of PCA components
n_neighbors = 5   # Number of nearest neighbors
n_clusters = 4   # Number of clusters


# Function to perform PCA, KMeans and identify single-item clusters
def identify_single_item_clusters(X, n_components, n_clusters):
    # Applying PCA
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(X_pca)


    # Identifying clusters with only one item
    unique, counts = np.unique(labels, return_counts=True)
    single_item_clusters = unique[counts == 1]

    # Finding the indices of these single-item clusters
    single_item_indices = np.where(np.isin(labels, single_item_clusters))[0]

    return single_item_indices, X_pca, labels


# Function to identify single-item clusters using Nearest Neighbors
def identify_single_item_clusters_nn(X, n_components, n_neighbors):
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
    threshold = np.mean(avg_distances) + 0.15 * np.std(avg_distances)
    single_item_indices = np.where(avg_distances > threshold)[0]

    return single_item_indices, X_pca

# Analyzing for each gender
single_item_clusters_info = {}
single_item_clusters_info_nn = {}


for gender in merged_df['Life_Expectancy_Variable'].unique():
    X = np.zeros((len(countries), len(drugs)))

    for i, drug in enumerate(drugs):
        # print(i, drug)
        info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
        filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]

        x = np.array(filter_gender['Pharma_Sales_Value'])
        X[:, i] = x


    # pca = PCA(n_components)
    # X_new = pca.fit_transform(X)
    # # print(np.shape(X_new))
    # labels = KMeans(n_clusters=n_clusters).fit_predict(X_new)
    # for i in np.unique(labels):
    #     plt.scatter(X_new[labels == i , 0], X_new[labels == i , 1], label = i)
    # # plt.scatter(X_new[:, 0], X_new[:, 1])
    # plt.legend()
    # plt.show()

    # # Identifying single-item clusters
    # single_item_indices, X_pca, labels = identify_single_item_clusters(X, n_components, n_clusters)
    # single_item_clusters_info[gender] = single_item_indices

    print(gender)

    # Identifying single-item clusters using nearest neighbors after PCA
    single_item_indices_nn, X_pca = identify_single_item_clusters_nn(X, n_components, n_neighbors)

    print(single_item_indices_nn)

    # scatterplot all dots without single_item_indices_nn
    for i in range(len(X_pca)):
        if i not in single_item_indices_nn:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], label=i, color='blue')
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], label=i, color='red')

    plt.show()

    single_item_clusters_info_nn[gender] = single_item_indices_nn


for gender in merged_df['Life_Expectancy_Variable'].unique():
    print(gender)

    for i, county in enumerate(countries):
        if gender in single_item_clusters_info_nn and i in single_item_clusters_info_nn[gender]:
            print(county)


