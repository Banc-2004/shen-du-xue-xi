import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据集
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("testA.csv")

# 数据预处理
floats_train = []
y_train_filtered = []
for signal, label in zip(df_train['heartbeat_signals'], df_train['label']):
    try:
        signal_list = signal.split(',')
        signal_array = np.array([float(x) for x in signal_list])
        if signal_array.ndim == 1:
            signal_array = signal_array.reshape(1, -1)
        floats_train.append(signal_array)
        y_train_filtered.append(int(label))  # 确保标签是整数
    except ValueError:
        continue

floats_train_array = np.vstack(floats_train)

# 标准化
scaler = StandardScaler()
floats_train_array_scaled = scaler.fit_transform(floats_train_array)

# 独热编码
num_classes = len(np.unique(y_train_filtered))
y_train_one_hot = np.eye(num_classes)[y_train_filtered]

# 定义模型
class CNNModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_features * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

num_features = floats_train_array_scaled.shape[1]
model = CNNModel(num_features, num_classes)

# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
X_train_tensor = torch.from_numpy(floats_train_array_scaled).float().unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train_one_hot.argmax(axis=1)).long()  # 使用argmax获取独热编码的索引

losses = []
accuracies = []
n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    losses.append(loss.item())
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_train_tensor).sum().item()
    accuracy = correct / len(y_train_tensor)
    accuracies.append(accuracy)
    loss.backward()
    optimizer.step()

# 测试模型
floats_test = []
for signal in df_test['heartbeat_signals']:
    try:
        signal_list = signal.split(',')
        signal_array = np.array([float(x) for x in signal_list])
        if signal_array.ndim == 1:
            signal_array = signal_array.reshape(1, -1)
        floats_test.append(signal_array)
    except ValueError:
        continue

floats_test_array = np.vstack(floats_test)
floats_test_array_scaled = scaler.transform(floats_test_array)
X_test_tensor = torch.from_numpy(floats_test_array_scaled).float().unsqueeze(1)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_probabilities = torch.softmax(test_outputs, dim=1).numpy()

# 保存结果
sample_submit = pd.read_csv("sample_submit.csv")
sample_submit[['label_0', 'label_1', 'label_2', 'label_3']] = test_probabilities
sample_submit.to_csv('sample_submit.csv', index=False)
print("完成")

# 可视化损失率和准确率
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.show()