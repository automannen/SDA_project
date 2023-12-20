import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# TODO: hier staat nog de KMeans clustering in en die hebben we nodig voor onze progressie plot. die er tot had geleid dat we Nearest Neighbors hebben gebruikt.

merged_df = pd.read_csv('merged_data.csv')
drugs = merged_df['Pharma_Sales_Variable'].unique()[1:]# skipping the total sales variable
countries = merged_df['Country'].unique()


# Directory for saving plots
plots_directory = '../data_visualization/clustering'
os.makedirs(plots_directory, exist_ok=True)


# Parameters
n_components = 2  # Number of PCA components
n_clusters = 4  # Number of clusters


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


# Analyzing for each gender
single_item_clusters_info = {}


for gender in merged_df['Life_Expectancy_Variable'].unique():
    X = np.zeros((len(countries), len(drugs)))

    for i, drug in enumerate(drugs):
        # print(i, drug)
        info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
        filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]

        x = np.array(filter_gender['Pharma_Sales_Value'])
        X[:, i] = x

    # Identifying single-item clusters
    single_item_indices, X_pca, labels = identify_single_item_clusters(X, n_components, n_clusters)
    single_item_clusters_info[gender] = single_item_indices

    print(gender)

    # scatterplot all dots without single_item_indices_nn
    for i in range(len(X_pca)):
        if i not in single_item_indices:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], label=i, color='blue')
        else:
            plt.scatter(X_pca[i, 0], X_pca[i, 1], label=i, color='red')


    # Some combination of drug-related features that contributed most to the variability in the data
    plt.xlabel('First principal component values')
    # Another composite measure/combination of features, capturing remaining variability
    plt.ylabel('Second principal component values')

    plt.legend()
    plt.title('Outliers KMeans ' + gender)

    plt_name = 'KNN_' + str(gender.replace(' ', '_')) + '.png'
    heatmap_path = os.path.join(plots_directory, plt_name)
    plt.savefig(heatmap_path)
    plt.close()




