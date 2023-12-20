import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# Read the merged data
merged_df = pd.read_csv('merged_data.csv')
drugs = merged_df['Pharma_Sales_Variable'].unique()[1:]  # Excluding the total sales variable
countries = merged_df['Country'].unique()

# Create directory for saving plots if it doesn't exist
plots_directory = '../data_visualization/clustering'
os.makedirs(plots_directory, exist_ok=True)

# Parameters for PCA and KMeans
n_components = 2  # Number of PCA components
n_clusters = 4  # Number of clusters

def identify_single_item_clusters(X, n_components, n_clusters):
    """Performs PCA and KMeans clustering to identify single-item clusters."""
    # Apply PCA
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    labels = kmeans.fit_predict(X_pca)

    # Identify clusters with only one item
    unique, counts = np.unique(labels, return_counts=True)
    single_item_clusters = unique[counts == 1]
    single_item_indices = np.where(np.isin(labels, single_item_clusters))[0]

    return single_item_indices, X_pca, labels

single_item_clusters_info = {}

for gender in merged_df['Life_Expectancy_Variable'].unique():
    # Create an array to store data for clustering
    X = np.zeros((len(countries), len(drugs)))
    for i, drug in enumerate(drugs):
        filtered_data = merged_df[(merged_df['Pharma_Sales_Variable'] == drug) &
                                  (merged_df['Life_Expectancy_Variable'] == gender)]
        X[:, i] = filtered_data['Pharma_Sales_Value'].values

    # Perform clustering
    single_item_indices, X_pca, labels = identify_single_item_clusters(X, n_components, n_clusters)
    single_item_clusters_info[gender] = single_item_indices

    plt.figure()
    for cluster_id in np.unique(labels):
        plt.scatter(X_pca[labels == cluster_id, 0], X_pca[labels == cluster_id, 1], label=cluster_id)

    plt.xlabel('First principal component values')
    plt.ylabel('Second principal component values')
    plt.legend()
    plt.title(f'Outliers KMeans {gender}')

    plt_name = f'KMeans_{gender.replace(" ", "_")}.png'
    heatmap_path = os.path.join(plots_directory, plt_name)
    plt.savefig(heatmap_path)
    plt.close()

