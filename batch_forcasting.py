import os
import torch
from transformers import AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from matplotlib import rcParams

# 设置字体
rcParams['font.family'] = 'Microsoft YaHei'  # 或 'SimHei'

# 设置负号正常显示
rcParams['axes.unicode_minus'] = False

# 文件夹路径
test_folder_path = 'dataset/1_year/cleaned/val'

# 读取所有CSV文件
csv_files = [f for f in os.listdir(test_folder_path) if f.endswith('.csv')]

# 加载模型
model = AutoModelForCausalLM.from_pretrained('logs/1y/fp16_5e-5cos_1024', device_map="cuda", trust_remote_code=True)

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

# 批量处理所有CSV文件
for csv_file in csv_files:
    # 读取数据
    data = pd.read_csv(os.path.join(test_folder_path, csv_file), header=None, parse_dates=[0])
    print(f"Processing file: {csv_file}")
    
    num_rows = data.shape[0]
    num_channels = data.shape[1] - 1  # 假设有多个通道
    print(f"数据总行数: {num_rows}")
    print(f"通道数: {num_channels}")

    # 设定预测长度
    prediction_length = 672  # 每次预测的长度

    # 设定历史数据长度
    history_length = 96  # 历史数据的长度

    # 设定窗口大小
    window_size = 96  # 滑动窗口的大小

    # 初始化预测的结果存储
    all_predictions = []
    all_true_values = []
    all_acc = []
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

        # 评估
        r2 = calculate_r2(test_tensor.cpu(), predictions.cpu())
        mse = mean_squared_error(test_tensor.cpu().numpy(), predictions.cpu().numpy())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_tensor.cpu().numpy(), predictions.cpu().numpy())
        mape = np.mean(np.abs((test_tensor.cpu().numpy() - predictions.cpu().numpy()) / test_tensor.cpu().numpy())) * 100
        ermse = calculate_ermse(test_tensor.cpu().numpy(), predictions.cpu().numpy())

        # 输出评估结果
        print(f"滑动窗口预测 {start_idx}-{start_idx+prediction_length} 的结果：")
        print(f"决定系数 (R²): {r2.item()}")
        print(f"均方误差 (MSE): {mse}")
        print(f"均方根误差 (RMSE): {rmse}")
        print(f"平均绝对误差 (MAE): {mae}")
        print(f"平均绝对百分比误差 (MAPE): {mape.item()}")
        print(f"预测准确率: {(1-ermse) * 100}%")

        # 保存预测和实际值
        all_acc.append((1-ermse) * 100)
        all_predictions.append(predictions.cpu().numpy())
        all_true_values.append(test_tensor.cpu().numpy())

    # 计算平均准确率
    avg_acc = np.mean(all_acc)
    print(f"全局平均准确率: {avg_acc}%")

    # 合并所有预测结果
    all_predictions = np.concatenate(all_predictions, axis=1)
    all_true_values = np.concatenate(all_true_values, axis=1)

    # 绘制实际值与预测值的对比图
    plt.figure(figsize=(32, 6))

    # 绘制每个通道的数据
    for channel in range(num_channels):
        plt.plot(range(count * prediction_length), all_true_values[channel], label=f'真实值 通道 {channel+1}', color='blue', linestyle='--')
        plt.plot(range(count * prediction_length), all_predictions[channel], label=f'预测值 通道{channel+1}', color='red')

    plt.legend()
    plt.xlabel('时间步（15分钟一个节点）')
    plt.ylabel('负荷')
    plt.title(f'{csv_file.replace(".csv", "")}预测结果\n预测准确率: {avg_acc}%')
    
    # 保存图像
    plt.savefig(f'预测结果_{csv_file.replace(".csv", "")}.png')
