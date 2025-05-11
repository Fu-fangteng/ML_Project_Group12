import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import silhouette_score


data = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")
X = data.iloc[:, 1:17].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
cluster = AgglomerativeClustering(n_clusters=7,  metric ='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(X_scaled)
silhouette = silhouette_score(X_scaled, cluster_labels)
print(f"silhouette_score: {silhouette:.3f}")



tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plot_df = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
plot_df['Cluster'] = cluster_labels
plot_df['Bean_Class'] = data.iloc[:, 0]
palette = sns.color_palette("husl", 7)  
scatter = sns.scatterplot(
    data=plot_df,
    x='t-SNE1',
    y='t-SNE2',
    hue='Cluster',
    palette=palette,
    style='Bean_Class', 
    s=100,
    edgecolor='black',
    alpha=0.8
)
# 添加图例和标题
plt.figure(figsize=(12, 8))
plt.title('Hierarchical_Clustering (t-SNE)', pad=20, fontsize=15)
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 添加网格
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()