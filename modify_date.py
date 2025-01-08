import pandas as pd
import numpy as np



# 读取 CSV 文件，更新文件路径去掉空格
file_path = 'dataset/5_year/cleaned/train/丽水全社会负荷.csv'
df = pd.read_csv(file_path, header=None)

# 检查是否存在 `<NULL>` 值
print("清理前的数据预览：")
print(df.head())
print("\n<NULL> 值计数：")
print((df == "<NULL>").sum())  # 统计每列 `<NULL>` 的个数

# 将 `<NULL>` 替换为 NaN
df.replace("<NULL>", float('nan'), inplace=True)
df.replace("#DIV/0!", float('nan'), inplace=True)

# 再次检查数据以确认 `<NULL>` 值替换成功
print("\n<NULL> 值替换为 NaN 后的数据预览：")
print(df.head())
print("\nNaN 值计数：")
print(df.isna().sum())  # 检查每列的 NaN 个数

# 删除包含 NaN 的行
df.dropna(inplace=True)

# 最终数据检查
print("\n删除 NaN 后的数据预览：")
print(df.head())
print("\n清理后的数据行数和列数：", df.shape)

# 保存清理后的数据
cleaned_file_path = "dataset/5_year/cleaned/train/丽水全社会负荷.csv"
df.to_csv(cleaned_file_path, index=False)
print(f"\n清理后的数据已保存至: {cleaned_file_path}")
