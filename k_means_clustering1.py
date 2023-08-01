# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:49:28 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\19th_august_k_clustring\2.K-MEANS CLUSTERING\Mall_Customers.csv')

x=dataset.iloc[:,[3,4]].values

#using the elbow method to find the optimal numbers for clusters
from sklearn.cluster import KMeans

wcss=[]

#to plot elbow method we have compute wcss for 10 diffrent number of cluster since we gonna have 10 iteration

#we are going to write for loop to create list 10 diffrent wcss for the 10 no of clusters
#thats why we have to initilaize wcss[],& start our loop

#we choose 1-11 becuase the 11 bound is excluded & we want 10 wcss however the first bound is included so hear i = 1,2,3 to 10
#now in each iteration of loop we are going to do 2 things  1st we are going to fit the k-means algorithm into our data x and we are going to compute WCSS
#Now lets fit kmean to our data x
#now eare 

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.scatter(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('no.of clusters')
plt.ylabel('wcss')
plt.show()
#wcss we have very good parameter called inertia_ credit goes to sklearn , that computes the sum of square , formula it will compute


#training the k-means model on dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans=kmeans.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
