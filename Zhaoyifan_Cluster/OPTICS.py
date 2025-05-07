import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/qiaoqian./Desktop/ML_Project/preprocessing/processed_data.csv')  # 请替换为你的绝对路径
X = df.iloc[:, 1:-7].values
y_true = df.iloc[:, -7:].values


# === 3. OPTICS 聚类 ===
optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
optics.fit(X)
labels = optics.labels_

# === 4. t-SNE 降维 ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# === 5. 可视化聚类结果 ===
plt.figure(figsize=(10, 5))

# 聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=20)
plt.title('OPTICS 聚类结果（t-SNE）')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

# 原始标签（可选）
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='tab10', s=20)
plt.title('原始标签分布（t-SNE）')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

plt.tight_layout()
plt.show()
