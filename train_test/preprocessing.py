import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 读取原始数据文件（请修改为你的文件路径）
data = pd.read_csv("processed_data_label_encoding.csv")  # 替换成你自己的文件名

# 2. 使用 train_test_split 划分数据（test_size=0.3 表示30%作为测试集）
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# 3. 保存训练集和测试集到新文件
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("划分完成，训练集和测试集已保存。")
