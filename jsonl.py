import pandas as pd
import json
import os

# 读取数据
data = pd.read_csv('dataset/3_year/cleaned/train/丽水全社会负荷.csv')

# 去掉表头并转置数据
data_transposed = data.drop(columns=['timestamp']).T

# 指定保存路径
output_dir = 'dataset/3_year/cleaned/train_jsonl'  # 请替换为你的文件夹路径
os.makedirs(output_dir, exist_ok=True)  # 确保文件夹存在
output_file = os.path.join(output_dir, '丽水全社会负荷.jsonl')

# 将数据转化为JSONL格式并保存
with open(output_file, 'w') as f:
    for index, row in data_transposed.iterrows():
        sequence = row.values.tolist()
        json_line = json.dumps({"sequence": sequence})
        f.write(json_line + '\n')

print(f"文件已保存至 {output_file}")
