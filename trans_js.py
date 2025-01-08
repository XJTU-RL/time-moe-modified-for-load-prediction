import os
import csv
import json
import pandas as pd

# 输入和输出文件夹路径
input_folder = 'dataset/1_year/cleaned/train/'  # 输入CSV文件所在的文件夹
output_folder = 'dataset/1_year/cleaned/train_jsonl/'  # 输出JSONL文件保存的文件夹

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取所有CSV文件
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# 批量转换CSV文件为JSONL格式
for csv_file in csv_files:
    csv_file_path = os.path.join(input_folder, csv_file)
    jsonl_file_path = os.path.join(output_folder, csv_file.replace('.csv', '.jsonl'))

    # 读取CSV文件并写入JSONL文件
    with open(csv_file_path, mode='r', encoding='utf-8') as infile, open(jsonl_file_path, mode='w', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)  # 使用DictReader将每一行转换为字典
        for row in reader:
            # 排除第一列（时间列）并将其他列的值转为浮动数值
            # 在此检查是否可以转换为浮动数值
            sequence = []
            for key, value in row.items():
                if key != 'date':  # 排除时间列
                    try:
                        sequence.append(float(value))  # 尝试转换为浮动数值
                    except ValueError:
                        continue  # 如果无法转换，跳过该列
            # 写入JSONL格式
            json_obj = {"sequence": sequence}
            outfile.write(json.dumps(json_obj) + '\n')

    print(f"已转换: {csv_file} -> {jsonl_file_path}")

# 读取并显示一个JSONL文件的前几行（示例）
sample_jsonl_file = os.path.join(output_folder, csv_files[0].replace('.csv', '.jsonl'))
df = pd.read_json(sample_jsonl_file, lines=True)
print(df.head())
