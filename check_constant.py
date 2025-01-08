import pandas as pd

# 读取CSV文件
df = pd.read_csv('dataset/3_year/cleaned/train/merged_data.csv')  # 你可以替换为其他文件

# 将 'date' 列转换为日期时间格式
df['date'] = pd.to_datetime(df['timestamp'])

# 计算时间间隔
df['time_diff'] = df['date'].diff().dt.total_seconds() / 60  # 时间差以分钟为单位

# 查找不连续部分
discontinuities = df[df['time_diff'] != 5]  # 5分钟为期望的时间间隔

# 输出不连续的时间戳和时间差
if len(discontinuities) > 0:
    print("不连续的时间戳及其间隔差异：")
    for index, row in discontinuities.iterrows():
        print(f"时间: {row['date']}, 前后间隔: {row['time_diff']} 分钟")
else:
    print("数据是连续的，每行之间的时间间隔为 5 分钟。")

# 计算不连续程度
discontinuity_count = len(discontinuities)
total_count = len(df)

# 输出不连续统计
print(f"\n数据总共 {total_count} 行，其中 {discontinuity_count} 行存在不连续。")
print(f"不连续程度为 {discontinuity_count / total_count * 100:.2f}%")