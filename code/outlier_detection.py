import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# TODO: verwijderen... KMeans clustering plot staat ook al in de "outlier_detection_olivier copy"

pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')
merged_df = pd.read_csv('merged_data.csv')
drugs = pharma_sales_df['Variable'].unique()[1:] # skipping the total sales variable
n_components = 2
n_clusters = 4

for gender in merged_df['Life_Expectancy_Variable'].unique():
    X = np.zeros((len(merged_df['Country'].unique()), len(drugs)))
    print(np.shape(X))

    for i, drug in enumerate(drugs):
        info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]

        filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]
        x = np.array(filter_gender['Pharma_Sales_Value'])
        X[:, i] = x

    pca = PCA(n_components)
    X_new = pca.fit_transform(X) # shape (33, 2), reduced to 2 dimensions
    labels = KMeans(n_clusters=n_clusters).fit_predict(X_new) # pakt de eerste outlier van eerste gender, tweede wel perfect
    print('labels', labels)
    test = np.unique(labels, return_counts=True)
    outlier_labels = np.unique(labels, return_counts=True)[0][np.unique(labels, return_counts=True)[1] < 2]
    print('cluster of outlier', outlier_labels)

    # Retrieving outlier from the original data
    for outlier_label in outlier_labels:
            outlier_indices = np.where(labels == outlier_label)[0]
            outlier_records = merged_df.iloc[outlier_indices]
            print(f"Outliers for {gender} in Cluster {outlier_label}:\n{outlier_records}")
    # aantal keer doen want hij assigned soms net andere cluster, soms perfect, soms net niet

    # Visualizing
    for i in np.unique(labels):
        plt.scatter(X_new[labels == i , 0], X_new[labels == i , 1], label = i)

    plt.legend()
    # Some combination of drug-related features that contributed most to the variability in the data
    plt.xlabel('First principal component values')
    # Another composite measure/combination of features, capturing remaining variability
    plt.ylabel('Second principal component values')
    plt.title(f'PCA followed by K-means cluster for {gender}')
    plt.show()

