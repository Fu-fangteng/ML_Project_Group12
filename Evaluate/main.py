import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
df = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")  
X = df.iloc[:, 1:17].values 
y_true = df.iloc[:, -1].values

kmeans = KMeans(n_clusters = 8, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components = 7, random_state=42)
labels_gmm = gmm.fit_predict(X)

hierarchical = AgglomerativeClustering(n_clusters = 6,  metric ='euclidean', linkage='ward')
labels_hierarchical = hierarchical.fit_predict(X)

print(X)
print(y_true)
print(labels_kmeans)
print(labels_gmm)
print(labels_hierarchical)