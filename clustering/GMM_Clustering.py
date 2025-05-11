import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE



data = pd.read_csv("C:/Users/lenovo/Desktop/ML Project/ML_Project_Group12/dry+bean+dataset/DryBeanDataset/processed_data.csv")
X = data.iloc[:, 1:17].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_components = np.arange(1, 11)
models = [GaussianMixture(n, random_state=42).fit(X_scaled) for n in n_components]
bic = [m.bic(X_scaled) for m in models]
aic = [m.aic(X_scaled) for m in models]

plt.figure(figsize=(10, 6))
plt.plot(n_components, bic, label='BIC')
plt.plot(n_components, aic, label='AIC')
plt.legend()
plt.xlabel('number of components')
plt.ylabel('信息准则值')
plt.title('Best components')
plt.xticks(n_components)
plt.grid(True)
plt.show()

# 选择最佳组件数（通常选择BIC最小的）
best_n = n_components[np.argmin(bic)]
print(f"suggested components: {best_n}")

# 使用最佳组件数拟合GMM
gmm = GaussianMixture(n_components=best_n, random_state=42)
gmm.fit(X_scaled)
labels = gmm.predict(X_scaled)

# 评估聚类质量
silhouette = silhouette_score(X_scaled, labels)
print(f"silhouette_score: {silhouette:.3f}")

# 降维可视化（使用t-SNE）
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab20', s=30, alpha=0.6)
plt.title(f'GMM_Clustering (k={best_n}, Silhouette Coefficient={silhouette:.2f})')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.colorbar(label='cluster labels')
plt.grid(True, alpha=0.3)
plt.show()