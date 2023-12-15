import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



pharma_sales_df = pd.read_csv('../data/pharma_sales_ppp.csv')
merged_df = pd.read_csv('merged_data.csv')
drugs = pharma_sales_df['Variable'].unique()[1:] # skipping the total sales variable
n_components = 2
n_clusters = 4

for gender in merged_df['Life_Expectancy_Variable'].unique():
    X = np.zeros((len(merged_df['Country'].unique()), len(drugs)))

    for i, drug in enumerate(drugs):
        info_drug_x = merged_df[merged_df['Pharma_Sales_Variable'] == drug][['Pharma_Sales_Variable', 'Pharma_Sales_Value', 'Life_Expectancy_Value', 'Life_Expectancy_Variable']]
        filter_gender = info_drug_x[info_drug_x['Life_Expectancy_Variable'] == gender]

        x = np.array(filter_gender['Pharma_Sales_Value'])
        X[:, i] = x

    pca = PCA(n_components)
    X_new = pca.fit_transform(X)
    # print(np.shape(X_new))
    labels = KMeans(n_clusters=n_clusters).fit_predict(X_new)
    for i in np.unique(labels):
        plt.scatter(X_new[labels == i , 0], X_new[labels == i , 1], label = i)
    # plt.scatter(X_new[:, 0], X_new[:, 1])
    plt.legend()
    plt.show()
    