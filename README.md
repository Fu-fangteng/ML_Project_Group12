# ML_Project_Group12


## 主要功能说明

- **数据预处理**  
  位于 [`preprocessing/`](preprocessing/) 目录，包括数据清洗、标准化、编码等操作。  
  - [`data_preprocessing.py`](preprocessing/data_preprocessing.py)：读取原始数据，处理缺失值、重复值，标准化特征并保存处理结果。
  - [`label_encoding.py`](preprocessing/label_encoding.py)：对类别标签进行编码。

- **数据集划分**  
  位于 [`train_test/`](train_test/) 目录。  
  - [`preprocessing.py`](train_test/preprocessing.py)：将预处理后的数据集划分为训练集和测试集。

- **聚类分析与评估**  
  位于 [`cluster&evaluation/`](cluster&evaluation/) 目录。  
  - [`main_cluster.py`](cluster&evaluation/main_cluster.py)：实现 KMeans、GMM、层次聚类等算法，并调用可视化模块展示聚类效果。
  - 各子目录（如 GMM、KMeans 等）存放对应聚类方法的实现细节。

- **可视化**  
  位于 [`visualization/`](visualization/) 目录。  
  - [`tsne_visualization.py`](visualization/tsne_visualization.py)：t-SNE 降维与聚类可视化。
  - [`visual_use.py`](visualization/visual_use.py)：可视化脚本调用示例。
  - `tsne_plots/`：保存 t-SNE 可视化结果。

- **数据**  
  - 原始数据集位于 [`data/DryBeanDataset/`](data/DryBeanDataset/)。

- **项目结构打印工具**  
  - [`structure.py`](structure.py)：可递归打印项目文件结构。

## 使用说明

1. **数据预处理**  
   运行 [`preprocessing/data_preprocessing.py`](preprocessing/data_preprocessing.py) 进行数据清洗和标准化。

2. **标签编码**  
   运行 [`preprocessing/label_encoding.py`](preprocessing/label_encoding.py) 进行标签编码。

3. **数据集划分**  
   运行 [`train_test/preprocessing.py`](train_test/preprocessing.py) 划分训练集和测试集。

4. **聚类分析与可视化**  
   运行 [`cluster&evaluation/main_cluster.py`](cluster&evaluation/main_cluster.py) 进行聚类分析，并可调用可视化模块展示结果。

5. **可视化**  
   运行 [`visualization/visual_use.py`](visualization/visual_use.py) 进行 t-SNE 可视化。

## 依赖环境

- Python 3.x
- pandas
- scikit-learn
- numpy
- 其它依赖请参考各脚本开头的 import 部分

---

