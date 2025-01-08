
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 数据，只读取第二列（value）
df = pd.read_csv('dataset/3_year/丽水全社会负荷_cleaned.csv', header=None, usecols=[1], names=['value'])

# 绘制第二列（value）的折线图
plt.figure(figsize=(10, 6))
plt.plot(df['value'], marker='o', linestyle='-', color='b', label='Value')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Value over Index')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 显示图形
plt.show()


# 保存图像到文件
plt.savefig('data_plot.png', dpi=300)

