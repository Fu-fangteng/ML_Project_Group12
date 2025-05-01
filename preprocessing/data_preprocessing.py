import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 设置文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(current_dir), 'dry+bean+dataset', 'DryBeanDataset')
input_file = os.path.join(data_dir, 'Dry_Bean_Dataset.xlsx')
output_file = os.path.join(current_dir, 'processed_data.csv')

# 读取数据
df = pd.read_excel(input_file)

# 检查缺失值
print("缺失值统计：")
print(df.isnull().sum())

# 检查重复数据
print("\n重复数据统计：")
# 检查整行重复
duplicates = df.duplicated()
print(f"整行完全重复的行数: {duplicates.sum()}")
if duplicates.sum() > 0:
    print("\n重复数据示例（显示两条）：")
    duplicate_rows = df[duplicates].head(2)
    print(duplicate_rows)
    
    # 显示这些重复行的原始行号
    print("\n这些重复行的原始行号：")
    for idx in duplicate_rows.index:
        print(f"行号 {idx}:")
        print(df.iloc[idx])
        print("-" * 50)

# 删除重复数据
df_cleaned = df.drop_duplicates()
print(f"\n删除重复数据后，数据集大小从 {len(df)} 减少到 {len(df_cleaned)}")

# 添加ID列
df_cleaned['ID'] = range(1, len(df_cleaned) + 1)
print("\n已添加ID列")

# 进行one-hot编码
# 假设'Class'是分类变量列
if 'Class' in df_cleaned.columns:
    # 对Class列进行one-hot编码
    df_encoded = pd.get_dummies(df_cleaned, columns=['Class'])
    print("\nOne-hot编码后的列名：")
    print(df_encoded.columns.tolist())

# 数据标准化
# 获取数值型列（排除ID列和one-hot编码后的列）
numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
if 'ID' in numeric_columns:
    numeric_columns = numeric_columns.drop('ID')

# 创建标准化器
scaler = StandardScaler()

# 对数值型列进行标准化
df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])
print("\n已完成数据标准化（使用StandardScaler）")
print("标准化后的数据统计：")
print(df_encoded[numeric_columns].describe())
    
# 保存处理后的数据
df_encoded.to_csv(output_file, index=False)
print(f"\n处理后的数据已保存到 {output_file}")

# 检查每列的重复值
print("\n各列重复值统计：")
for column in df.columns:
    duplicate_count = df[column].duplicated().sum()
    print(f"{column}: {duplicate_count} 个重复值")





