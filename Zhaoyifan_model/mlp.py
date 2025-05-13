import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



# 将数据转换为 PyTorch 的张量（如果数据是 NumPy 数组的话）
X_tensor = torch.tensor(X)  # 数据矩阵 (num_samples, num_features)
y_tensor = torch.tensor(y)  # 标签向量 (num_samples)

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# 超参数
input_size = X_tensor.shape[1]  # 输入特征数
hidden_size = 64                # 隐藏层大小
output_size = len(torch.unique(y_tensor))  # 输出类别数

# 实例化模型
model = MLP(input_size, hidden_size, output_size)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 后向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重

        running_loss += loss.item()

    # 每个 epoch 输出损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 测试模型（示例）
model.eval()
with torch.no_grad():
    # 使用训练数据进行预测（示例）
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == y_tensor).float().mean()
    print(f"Accuracy: {accuracy:.4f}")
