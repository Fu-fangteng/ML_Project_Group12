import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('/Users/qiaoqian./Desktop/ML_Project/preprocessing/processed_data_label_encoding.csv')  # 请替换为你的绝对路径
X = df.iloc[:, 1:17].values  # 特征
y_true = df.iloc[:, -1].values  # 原始标签

# === 3. K-means 聚类 ===
kmeans = KMeans(n_clusters=7, random_state=42)  # 可以根据需要选择不同的簇数
labels = kmeans.fit_predict(X)  # 获取聚类标签

# 打印每个类别的数量
unique_labels, counts = np.unique(labels, return_counts=True)

# 显示聚类类别及其数量
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} points")

# === 4. t-SNE 降维 ===
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# === 5. 可视化聚类结果 ===
plt.figure(figsize=(10, 5))

# 聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', s=20)
plt.title('K-means Cluster (t-SNE)')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

# 原始标签（可选）
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='tab10', s=20)
plt.title('Original (t-SNE)')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')

plt.tight_layout()
plt.show()
