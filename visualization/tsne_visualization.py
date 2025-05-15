import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px


def tsne_visualize(data_file, output_dir='tsne_plots', perplexity=30, n_iter=1000, random_state=42):

    # 读取数据
    df = pd.read_csv(data_file)

    # 特征列（排除 ID 和标签列）
    feature_columns = [col for col in df.columns if col not in ['ID', 'encoded_label']]
    X = df[feature_columns].values

    # 标签列
    if 'encoded_label' in df.columns:
        y = df['encoded_label'].astype(str).values  # 转为字符串以便可视化上色
    else:
        raise ValueError("数据中未找到 'encoded_label' 列。请检查输入数据格式。")

    # t-SNE 降维
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    X_embedded = tsne.fit_transform(X)

    # 构造 DataFrame
    df_embedded = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df_embedded['Class'] = y

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 2D 可视化
    fig_2d = px.scatter(df_embedded, x='TSNE1', y='TSNE2', color='Class',
                        title='t-SNE 2D Interactive Visualization', width=900, height=700)
    fig_2d.update_traces(marker=dict(size=4))  # 控制点大小
    fig_2d.write_html(os.path.join(output_dir, 'tsne_2d_interactive.html'))

    # 3D 可视化
    fig_3d = px.scatter_3d(df_embedded, x='TSNE1', y='TSNE2', z='TSNE3', color='Class',
                           title='t-SNE 3D Interactive Visualization', width=900, height=700)
    fig_3d.update_traces(marker=dict(size=2))  # 控制点大小
    fig_3d.write_html(os.path.join(output_dir, 'tsne_3d_interactive.html'))

    print(f"交互式可视化图已保存至：{output_dir}")


def tsne_cluster_visualize(X, labels, output_dir='tsne_cluster_plots', name='kmeans', perplexity=30, max_iter=1000, random_state=42):
    """
    对聚类结果进行t-SNE降维并可视化。
    """
    print(f"执行 t-SNE 降维并可视化: {name}")

    # t-SNE
    tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=max_iter, random_state=random_state)
    X_embedded = tsne.fit_transform(X)

    df_embedded = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df_embedded['Cluster'] = labels.astype(str)  # 转为字符串用于颜色分组

    # 创建目录
    os.makedirs(output_dir, exist_ok=True)

    # 2D 可视化
    fig_2d = px.scatter(df_embedded, x='TSNE1', y='TSNE2', color='Cluster',
                        title=f't-SNE 2D Clustering Visualization - {name}', width=900, height=700)
    fig_2d.update_traces(marker=dict(size=4))  # 控制点大小
    fig_2d.write_html(os.path.join(output_dir, f'{name}_tsne_2d.html'))

    # 3D 可视化
    fig_3d = px.scatter_3d(df_embedded, x='TSNE1', y='TSNE2', z='TSNE3', color='Cluster',
                           title=f't-SNE 3D Clustering Visualization - {name}', width=900, height=700)
    fig_3d.update_traces(marker=dict(size=2))  # 控制点大小
    fig_3d.write_html(os.path.join(output_dir, f'{name}_tsne_3d.html'))

    print(f"{name} 聚类可视化图已保存至：{output_dir}")
