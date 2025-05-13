import pandas as pd


# 1. 加载数据
data = pd.read_csv('processed_data_onehot.csv')  # 将 'your_file.csv' 替换为你的文件路径

# 2. 获取one-hot编码的标签列（假设最后7列是one-hot编码的标签）
onehot_labels = data.iloc[:, -7:].to_numpy()  # 取最后7列作为one-hot标签


# 一般来说，one-hot编码的每一列代表一个类别，进行标签编码时需要找到最大类别索引
y_true = onehot_labels.argmax(axis=1)  # 从one-hot编码中获取对应的整数标签

data['encoded_label'] = y_true 
print(data.head())

data.drop(data.columns[-8:-1], axis=1, inplace=True) 
# 查看转换后的数据
print(data.head())

# 你可以选择将新的标签编码保存为csv文件
data.to_csv('processed_data_label_encoding.csv', index = False)
