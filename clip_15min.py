import os
import pandas as pd

# 设定输入输出文件夹路径
input_folder = 'dataset/5_year'  # 输入文件夹路径
output_folder = 'dataset/5_year'  # 输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        
        # 读取数据文件
        df = pd.read_csv(file_path, header=None)

        # 根据文件的列数自动设置列名
        columns = ['timestamp'] + [f'col{i}' for i in range(2, len(df.columns) + 1)]
        df.columns = columns

        # 将timestamp列转换为datetime格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 设置timestamp为索引
        df.set_index('timestamp', inplace=True)

        # 重新采样，按15分钟进行重采样，选择每15分钟的第一个数据点
        df_resampled = df.resample('15T').first()

        # 保留每小时的00、15、30、45分钟数据点
        df_filtered = df_resampled[df_resampled.index.minute.isin([0, 15, 30, 45])]

        # 保存为新的CSV文件
        output_file_path = os.path.join(output_folder, filename)
        df_filtered.to_csv(output_file_path)

        print(f"Processed {filename}")
