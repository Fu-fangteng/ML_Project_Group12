import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px


def tsne_visualize(data_file, output_dir='tsne_plots', perplexity=30, n_iter=1000, random_state=42):
    """
    对指定CSV文件执行t-SNE降维，并生成交互式2D和3D可视化图。

    参数:
        data_file (str): 预处理后CSV文件路径
        output_dir (str): 输出交互图的HTML保存路径
        perplexity (int): t-SNE中的perplexity参数
        n_iter (int): t-SNE迭代次数
        random_state (int): 随机种子

    输出:
        HTML交互式图像保存在output_dir中
    """
    # 读取数据
    df = pd.read_csv(data_file)

    # 特征和标签处理
    feature_columns = df.iloc[:, 1:17].values
    class_columns = df.iloc[:, -1].values
    X = df[feature_columns].values
    y = df[class_columns].values

    # t-SNE 降维
    print("执行t-SNE降维...")
    tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=n_iter, random_state=random_state)
    X_embedded = tsne.fit_transform(X)

    # 结果DataFrame
    df_embedded = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2', 'TSNE3'])
    df_embedded['Class'] = y.astype(str)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 2D 可视化
    fig_2d = px.scatter(df_embedded, x='TSNE1', y='TSNE2', color='Class',
                        title='t-SNE 2D Interactive Visualization', width=900, height=700)
    fig_2d.write_html(os.path.join(output_dir, 'tsne_2d_interactive.html'))

    # 3D 可视化
    fig_3d = px.scatter_3d(df_embedded, x='TSNE1', y='TSNE2', z='TSNE3', color='Class',
                           title='t-SNE 3D Interactive Visualization', width=900, height=700)
    fig_3d.write_html(os.path.join(output_dir, 'tsne_3d_interactive.html'))

    print(f"交互式可视化图已保存至：{output_dir}")