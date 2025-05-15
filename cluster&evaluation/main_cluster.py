import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
import sys
import os

# 添加上一级目录（即 project 根目录）到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualization.tsne_visualization import tsne_cluster_visualize


df = pd.read_csv(r"..\preprocessing\processed_data_label_encoding.csv")  
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
# print(labels_gmm)
# print(labels_hierarchical)




tsne_cluster_visualize(X, labels_kmeans, output_dir='clustering_plots', name='KMeans')
tsne_cluster_visualize(X, labels_gmm, output_dir='clustering_plots', name='GMM')
tsne_cluster_visualize(X, labels_hierarchical, output_dir='clustering_plots', name='Hierarchical')






