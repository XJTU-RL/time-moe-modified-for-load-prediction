import torch
from transformers import AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns

# 读取数据
data = pd.read_csv('dataset/5_year/cleaned/val/merged_data.csv', header=0, parse_dates=[0])
print(f"Data shape: {data.shape}")
num_rows = data.shape[0]
print(f"数据总行数: {num_rows}")

# 判断数据的通道数
num_channels = data.shape[1] - 1  # 假设有多个通道，如果数据包含更多通道，需根据实际情况修改
print(f"通道数: {num_channels}")

# 设定预测长度
prediction_length = 96  # 每次预测的长度

# 设定历史数据长度
history_length = 672  # 历史数据的长度

# 设定窗口大小
window_size = 12  # 滑动窗口的大小

# 加载模型
model = AutoModelForCausalLM.from_pretrained('logs/5y/fp16_5e-5cos_1024/checkpoint-92', device_map="cuda", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-200M', device_map="cuda", trust_remote_code=True)

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # 将模型移动到设备

# 评估指标函数
def calculate_r2(true_values, predicted_values):
    ss_total = torch.sum((true_values - torch.mean(true_values))**2)
    ss_residual = torch.sum((true_values - predicted_values)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def calculate_ermse(y_true, y_pred):
    mse = np.mean(((y_true - y_pred) / y_true) ** 2)
    ermse = np.sqrt(mse)
    return ermse

# 初始化预测的结果存储
all_predictions = []
all_true_values = []
all_acc = []
all_acc_l = []
count = 0
# 使用滑动窗口进行预测
for start_idx in range(0, num_rows - history_length - prediction_length, window_size):
    count += 1
    # 提取当前窗口的数据
    seqs = pd.concat([data.iloc[:, i+1][start_idx: start_idx + history_length] for i in range(num_channels)], axis=1)
    # 转换为Tensor并调整维度
    seqs_tensor = torch.tensor(seqs.values, dtype=torch.float32).transpose(0, 1)
    seqs_tensor = seqs_tensor.to(device)  # 移动到设备
    
    # 标准化
    mean, std = seqs_tensor.mean(dim=-1, keepdim=True), seqs_tensor.std(dim=-1, keepdim=True)
    normed_seqs = (seqs_tensor - mean) / std
    
    # 预测
    output = model.generate(normed_seqs, max_new_tokens=prediction_length)
    normed_predictions = output[:, -prediction_length:]
    
    # 反向标准化
    predictions = normed_predictions * std + mean

    # 提取对应的测试数据
    test = pd.concat([data.iloc[ : , i + 1][start_idx + history_length: start_idx + history_length + prediction_length] for i in range(num_channels)], axis=1)
    test_tensor = torch.tensor(test.values, dtype=torch.float32).transpose(0, 1)
    # print(f"test_tensor.shape: {test_tensor.shape}")

    # 评估
    # r2 = calculate_r2(test_tensor.cpu(), predictions.cpu())
    # mse = mean_squared_error(test_tensor.cpu().numpy(), predictions.cpu().numpy())
    # rmse = np.sqrt(mse)
    # mae = mean_absolute_error(test_tensor.cpu().numpy(), predictions.cpu().numpy())
    # mape = np.mean(np.abs((test_tensor.cpu().numpy() - predictions.cpu().numpy()) / test_tensor.cpu().numpy())) * 100
    ermse = calculate_ermse(test_tensor.cpu().numpy(), predictions.cpu().numpy())

    # # 输出评估结果
    print(f"滑动窗口预测 {start_idx + history_length}-{start_idx + history_length+prediction_length} 的结果：")
    # print(f"决定系数 (R²): {r2.item()}")
    # print(f"均方误差 (MSE): {mse}")
    # print(f"均方根误差 (RMSE): {rmse}")
    # print(f"平均绝对误差 (MAE): {mae}")
    # print(f"平均绝对百分比误差 (MAPE): {mape.item()}")
    print(f"单步预测准确率: {(1-ermse) * 100}%")

    # 保存预测和实际值，并对预测出的坏数据进行过滤
    lv = 0.85
    if 1-ermse > lv:
        all_acc_l.append((1-ermse) * 100)
    all_acc.append((1-ermse) * 100)
    print(f"全局平均准确率: {np.mean(all_acc)}%\n去除{(lv * 100)}%以下结果后的全局平均准确率: {np.mean(all_acc_l)}%")
    all_predictions.append(predictions.cpu().numpy())
    all_true_values.append(test_tensor.cpu().numpy())

# 计算平均准确率
avg_acc = np.mean(all_acc)
print(f"全局平均准确率: {avg_acc}%")
avg_acc_l = np.mean(all_acc_l)
print(f"去除{(lv * 100)}%以下结果后的全局平均准确率: {avg_acc_l}%")

# 合并所有预测结果
all_predictions = np.concatenate(all_predictions, axis=1)
all_true_values = np.concatenate(all_true_values, axis=1)


# 绘制实际值与预测值的对比图
plt.figure(figsize=(32, 6))

# 绘制每个通道的数据
for channel in range(num_channels):
    plt.plot(range(count * prediction_length), all_true_values[channel], label=f'Actual - Channel {channel+1}', color='blue', linestyle='--')
    plt.plot(range(count * prediction_length), all_predictions[channel], label=f'Prediction - Channel {channel+1}', color='red')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'Prediction vs Actual Values for All Channels\naverage accuracy: {avg_acc}%')
plt.savefig('滑动窗口预测结果.png')

# 假设你的预测结果保存在一个列表中
predictions = all_acc  # 替换为你的预测结果列表

# 设置 Seaborn 样式
sns.set(style="whitegrid")

# 创建一个图形
plt.figure(figsize=(10, 6))

# 使用 Seaborn 绘制分布图
sns.histplot(predictions, kde=True, bins=100)

# 设置图表标题和标签
plt.title("Prediction Results Distribution")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")

# 保存图形到文件
plt.savefig("prediction_distribution.png", dpi=300, bbox_inches='tight')

# 假设你的预测结果保存在一个列表中
predictions = all_acc_l  # 替换为你的预测结果列表

# 设置 Seaborn 样式
sns.set(style="whitegrid")

# 创建一个图形
plt.figure(figsize=(10, 6))

# 使用 Seaborn 绘制分布图
sns.histplot(predictions, kde=True, bins=100)

# 设置图表标题和标签
plt.title("Prediction Results Distribution")
plt.xlabel("Predicted Value")
plt.ylabel("Frequency")

# 保存图形到文件
plt.savefig("prediction_distribution_l.png", dpi=300, bbox_inches='tight')
