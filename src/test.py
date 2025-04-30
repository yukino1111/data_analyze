import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from common import *

data = pd.read_csv("./assets/Student_performance_data.csv")
data = clean_wash(data)
# 绘制指定列的分布图
columns_to_plot = ["StudyTimeWeekly", "Absences", "Age"]

# 绘制 ParentalEducation 和 ParentalInvolvement 的分布图
plt.figure(figsize=(12, 6))  # 调整整体图形大小
# ParentalEducation 子图
plt.subplot(1, 2, 1)  # 1 行 2 列，第一个子图
sns.countplot(data=data, x="ParentalEducation")  # 绘制计数图
plt.title("ParentalEducation 的分布")
plt.xlabel("ParentalEducation")
plt.ylabel("频率")
plt.xticks(range(5))  # 设置 x 轴刻度
# ParentalInvolvement 子图
plt.subplot(1, 2, 2)  # 1 行 2 列，第二个子图
sns.countplot(data=data, x="ParentalInvolvement")  # 绘制计数图
plt.title("ParentalInvolvement 的分布")
plt.xlabel("ParentalInvolvement")
plt.ylabel("频率")
plt.xticks(range(5))  # 设置 x 轴刻度
plt.tight_layout()  # 自动调整子图参数，避免重叠
plt.savefig("./assets/pic/Parental_Distribution.png")
plt.show()

