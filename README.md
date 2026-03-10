# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select Annual Income and Spending Score.
2. Use the Elbow Method to find the optimal number of clusters.
3. Apply K-Means clustering with the chosen number of clusters (k = 5).
4. Plot the clusters and centroids to visualize customer groups.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Ashwin Kumar .M
RegisterNumber:  212225040033
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")
print(data.head())

X = data.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='yellow', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='black', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='red', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=300, c='blue', label='Centroids')

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```


## Output:
<img width="791" height="150" alt="image" src="https://github.com/user-attachments/assets/77e1c84e-c96c-4856-95a3-f68485864ee3" />
<img width="965" height="593" alt="image" src="https://github.com/user-attachments/assets/c0614fc3-68c7-4354-89be-0859323ed7cc" />
<img width="801" height="564" alt="image" src="https://github.com/user-attachments/assets/c59b9f28-b2d9-4533-bae2-156b255515e1" />




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
