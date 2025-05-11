import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# 读取数据
df = pd.read_csv('/Users/qiaoqian./Desktop/ML_Project/preprocessing/processed_data_label_encoding.csv')
X = df.iloc[:, 1:17].values
y_true = df.iloc[:, -1].values

# === 聚类 ===
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X)

# 打印每个类别的数量
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} points")

# === 外部指标 ===
print("\n--- External Evaluation Metrics ---")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(y_true, labels):.4f}")
print(f"Normalized Mutual Information (NMI): {normalized_mutual_info_score(y_true, labels):.4f}")
print(f"Fowlkes-Mallows Index (FMI): {fowlkes_mallows_score(y_true, labels):.4f}")

# === 内部指标 ===
print("\n--- Internal Evaluation Metrics ---")
print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X, labels):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.4f}")

# === 降维可视化 ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(10, 5))

# 聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=20)
plt.title('K-means Cluster (t-SNE)')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

# 原始标签
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='tab10', s=20)
plt.title('Original (t-SNE)')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

plt.tight_layout()
plt.show()
