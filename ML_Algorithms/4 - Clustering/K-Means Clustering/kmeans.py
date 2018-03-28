# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Using elbow method 
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmean = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmean.fit(X)
    wcss.append(kmean.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-Means to the dataset
kmean = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmean = kmean.fit_predict(X)

# Visualising the results
plt.scatter(X[y_kmean == 0,0], X[y_kmean == 0, 1], s = 100 ,c ='red' , label = 'Careful')
plt.scatter(X[y_kmean == 1,0], X[y_kmean == 1, 1], s = 100 ,c ='blue' , label = 'Standard')
plt.scatter(X[y_kmean == 2,0], X[y_kmean == 2, 1], s = 100 ,c ='green' , label = 'Target')
plt.scatter(X[y_kmean == 3,0], X[y_kmean == 3, 1], s = 100 ,c ='cyan' , label = 'Careless')
plt.scatter(X[y_kmean == 4,0], X[y_kmean == 4, 1], s = 100 ,c ='magenta' , label = 'Sensible')
plt.scatter(kmean.cluster_centers_[:, 0],kmean.cluster_centers_[:, 1], s = 300 ,c ='yellow' , label = 'Centroid')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()