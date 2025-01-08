import os
import pandas as pd

# 文件夹路径
folder_path = 'dataset/1_year/'

# 获取文件夹中所有文件的文件名
files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 创建一个空的DataFrame来存储合并后的数据
merged_data = pd.DataFrame()

# 遍历每个文件
for file in files:
    file_path = os.path.join(folder_path, file)
    
    # 读取文件
    data = pd.read_csv(file_path, header=0, names=["timestamp", file])
    
    # 将timestamp列转换为datetime格式
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # 合并数据，基于timestamp列
    if merged_data.empty:
        merged_data = data
    else:
        merged_data = pd.merge(merged_data, data, on="timestamp", how="inner")

# 将时间戳设置为索引
merged_data.set_index('timestamp', inplace=True)

# 保存合并后的数据为CSV文件
merged_data.to_csv('dataset/1_year/merged_data.csv')

print("数据已成功合并并保存为 'merged_data.csv'.")
