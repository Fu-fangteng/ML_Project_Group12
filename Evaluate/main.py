import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from Evaluate.Kmeans_evaluate import optimal_n_kmeans
from Evaluate.GMM_evaluate import optimal_n_gmm
from Evaluate.Hierarchical_evaluate import optimal_n_hierarchical
df = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")  
X = df.iloc[:, 1:17].values 
y_true = df.iloc[:, -1].values

kmeans = KMeans(n_clusters = optimal_n_kmeans, random_state=42)
labels_kmeans = kmeans.fit_predict(X)

gmm = GaussianMixture(n_clusters = optimal_n_gmm, random_state=42)
labels_gmm = gmm.fit_predict(X)

hierarchical = AgglomerativeClustering(n_clusters = optimal_n_hierarchical,  metric ='euclidean', linkage='ward')
labels_hierarchical = hierarchical.fit_predict(X)
