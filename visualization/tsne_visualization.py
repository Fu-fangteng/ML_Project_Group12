import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocessing_dir = os.path.join(os.path.dirname(current_dir), 'preprocessing')
data_file = os.path.join(preprocessing_dir, 'processed_data.csv')

# 读取处理后的数据
df = pd.read_csv(data_file)

# 获取特征列（排除ID列和Class相关的列）
feature_columns = [col for col in df.columns if col != 'ID' and not col.startswith('Class_')]
X = df[feature_columns].values

# 获取类别标签
class_columns = [col for col in df.columns if col.startswith('Class_')]
y = df[class_columns].values.argmax(axis=1)  # 将one-hot编码转换回类别索引

# 执行t-SNE降维
print("执行t-SNE降维...")
tsne_2d = TSNE(n_components=2, random_state=42)
X_2d = tsne_2d.fit_transform(X)

tsne_3d = TSNE(n_components=3, random_state=42)
X_3d = tsne_3d.fit_transform(X)

# 创建可视化目录
output_dir = os.path.join(current_dir, 'tsne_plots')
os.makedirs(output_dir, exist_ok=True)

# 2D可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE 2D Visualization')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig(os.path.join(output_dir, 'tsne_2d.png'))
plt.close()

# 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
ax.set_title('t-SNE 3D Visualization')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')
plt.savefig(os.path.join(output_dir, 'tsne_3d.png'))
plt.close()

print(f"可视化结果已保存到 {output_dir} 目录") 