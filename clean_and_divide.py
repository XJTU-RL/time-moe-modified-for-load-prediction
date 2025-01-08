import os
import pandas as pd
import numpy as np

# 设置文件夹路径和输出路径
input_folder = "dataset/1_year/"
output_folder = "dataset/1_year/cleaned/"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)

# 获取文件夹中的所有文件
files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# 处理每个文件
for file in files:
    file_path = os.path.join(input_folder, file)
    df = pd.read_csv(file_path, header=0)
    
    # 检查并处理 `<NULL>` 和 `#DIV/0!`
    print(f"清理文件: {file}")
    df.replace("<NULL>", float('nan'), inplace=True)
    df.replace("#DIV/0!", float('nan'), inplace=True)
    
    # 删除包含 NaN 的行
    df.dropna(inplace=True)
    
    # 计算分割位置
    total_rows = len(df)
    train_size = int(total_rows * 0.7)  # 70% 用于训练集
    test_size = int(total_rows * 0.15)  # 15% 用于测试集
    val_size = total_rows - train_size - test_size  # 剩余 15% 用于验证集

    # 划分数据集：按顺序划分
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:train_size + test_size]
    val_df = df.iloc[train_size + test_size:]
    
    # 保存清理后的数据到不同的文件夹
    train_file_path = os.path.join(output_folder, 'train', f"{file}")
    test_file_path = os.path.join(output_folder, 'test', f"{file}")
    val_file_path = os.path.join(output_folder, 'val', f"{file}")
    
    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)
    val_df.to_csv(val_file_path, index=False)
    
    print(f"文件已保存至训练集: {train_file_path}")
    print(f"文件已保存至测试集: {test_file_path}")
    print(f"文件已保存至验证集: {val_file_path}")

print("\n所有文件处理完毕！")
