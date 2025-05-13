import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
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
from Cluster_evaluation import Cluster_Evaluation
from interactive_visualization import tsne_visualize

df = pd.read_csv('/Users/qiaoqian./Desktop/ML_Project/preprocessing/processed_data_label_encoding.csv')  # 请替换为你的绝对路径
X = df.iloc[:, 1:17].values
y_true = df.iloc[:, -1].values

# === 3. OPTICS 聚类 ===
optics = OPTICS(min_samples=5, xi=0.1, min_cluster_size=0.05)
optics.fit(X)
labels = optics.labels_

# 打印每个类别的数量
unique_labels, counts = np.unique(labels, return_counts=True)

# 显示聚类类别及其数量
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} points")

tsne_visualize(data_file="/Users/qiaoqian./Desktop/ML_Project/preprocessing/processed_data_label_encoding.csv")

# === 4. t-SNE 降维 ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)


# === 5. 可视化聚类结果 ===
plt.figure(figsize=(10, 5))

# 聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=20)
plt.title('OPTICS Cluste（t-SNE）')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

# 原始标签（可选）
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='tab10', s=20)
plt.title('Original（t-SNE）')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

plt.tight_layout()
plt.show()
