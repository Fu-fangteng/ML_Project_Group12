from tsne_visualization import tsne_visualize
from tsne_visualization import tsne_cluster_visualize



# 替换成你实际的 CSV 文件路径
data_path = '../preprocessing/processed_data_label_encoding.csv'
data_output = r"D:\大二spring\ML\project\visualization"
tsne_visualize(data_file=data_path,output_dir=data_output)

