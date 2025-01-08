import torch
from transformers import AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 读取数据
data = pd.read_csv('丽水气象电力数据/丽水龙泉市预测/测试数据2.csv', header=None, parse_dates=[0])
num_rows = data.shape[0]
print(f"Number of rows: {num_rows}")
print(data.shape)
# 判断数据的通道数
num_channels = data.shape[1] - 1  # 假设有两个通道，如果数据包含更多通道，需修改这里
print(f"Number of channels: {num_channels}")

# 设定预测长度
prediction_length = 96  # 预测长度

# 抽取数据作为训练数据
seqs = pd.concat([data.iloc[:, i+1][:-prediction_length] for i in range(num_channels)], axis=1)

# 转换为Tensor并调整维度 (通道数, 序列长度)
seqs_tensor = torch.tensor(seqs.values, dtype=torch.float32).transpose(0, 1)

# 抽取数据作为测试数据
test = pd.concat([data.iloc[:, i+1][-prediction_length :] for i in range(num_channels)], axis=1)
test_tensor = torch.tensor(test.values, dtype=torch.float32).transpose(0, 1)

# 输出训练集和测试集的形状
print(f"Training shape: {seqs_tensor.shape}, Test shape: {test_tensor.shape}")

# 加载模型
model = AutoModelForCausalLM.from_pretrained('logs/time_moe', device_map="cuda", trust_remote_code=True)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到设备
seqs_tensor = seqs_tensor.to(device)  # 将输入数据移动到设备

# 标准化
mean, std = seqs_tensor.mean(dim=-1, keepdim=True), seqs_tensor.std(dim=-1, keepdim=True)
normed_seqs = (seqs_tensor - mean) / std


print(f"Using device: {device}")


# 预测
output = model.generate(normed_seqs, max_new_tokens=prediction_length)
normed_predictions = output[:, -prediction_length:]
print(f"Predictions shape: {normed_predictions.shape}")


# 反向标准化
predictions = normed_predictions * std + mean

test_tensor = test_tensor.cpu()
predictions = predictions.cpu()




mse = mean_squared_error(test_tensor.numpy().flatten(), predictions.numpy().flatten())  # 均方误差
rmse = np.sqrt(mse)  # 均方根误差
mae = mean_absolute_error(test_tensor.numpy().flatten(), predictions.numpy().flatten())  # 平均绝对误差

print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")

# 绘制所有通道的实际值与预测值对比图
plt.figure(figsize=(32, 6))

# 绘制每个通道的数据
for channel in range(num_channels):
    # plt.plot(range(num_rows - prediction_length), seqs_tensor[channel], label=f'Train Actual - Channel {channel+1}', color='blue')
    plt.plot(range(num_rows - prediction_length, num_rows), test_tensor[channel], label=f'Test Actual - Channel {channel+1}', color='blue', linestyle='--')
    plt.plot(range(num_rows - prediction_length, num_rows), predictions[channel], label=f'Predicted - Channel {channel+1}', color='red')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Prediction vs Actual Values for All Channels')
plt.show()
