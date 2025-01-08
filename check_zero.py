import pandas as pd

# 读取数据
data = pd.read_csv('dataset/3_year/cleaned/train/merged_data.csv')

# 去掉 timestamp 列，检查其他列是否有零值
zero_values = data.drop(columns=['timestamp']) == 0

# 输出包含零值的行
rows_with_zeros = zero_values.any(axis=1)

if rows_with_zeros.any():
    print("存在零值的行：")
    print(data[rows_with_zeros])
else:
    print("没有零值")
