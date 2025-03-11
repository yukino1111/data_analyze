import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 可选，用于更美观的图表

# 1. 读取数据
# 假设你的数据保存在一个名为 "student_data.csv" 的文件中
data = pd.read_csv("./assets/Student_performance_data _.csv")

# 2. 数据预处理 (可选，但通常很有用)
#   - 检查缺失值
#   - 转换数据类型 (例如，将分类变量转换为数值型)
#   - 创建新的特征 (如果需要)

# 检查缺失值
print(data.isnull().sum())  # 打印每列缺失值的数量

# 如果有缺失值，可以选择填充或删除
# 例如，用平均值填充 GPA 列的缺失值：
# data['GPA'].fillna(data['GPA'].mean(), inplace=True)

# 将分类变量转换为数值型 (如果需要)
# 例如，将 Gender 列转换为 0 和 1：
# data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})  # 如果 Gender 列是字符串

# 3. 分析 GPA 与其他因素的关系

# a.  数值型变量与 GPA 的关系 (使用散点图或相关系数)
numerical_features = [
    "Age",
    "StudyTimeWeekly",
    "Absences",
    "ParentalSupport",
]  # 添加你认为相关的数值型特征

for feature in numerical_features:
    plt.figure(figsize=(8, 6))  # 设置图表大小
    plt.scatter(data[feature], data["GPA"])
    plt.xlabel(feature)
    plt.ylabel("GPA")
    plt.title(f"GPA vs. {feature}")
    plt.show()

    # 计算相关系数
    correlation = data[feature].corr(data["GPA"])
    print(f"{feature} 与 GPA 的相关系数: {correlation}")

# b.  类别型变量与 GPA 的关系 (使用箱线图或条形图)
categorical_features = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]  # 添加你认为相关的类别型特征

for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=feature, y="GPA", data=data)  # 使用 seaborn 的箱线图
    plt.xlabel(feature)
    plt.ylabel("GPA")
    plt.title(f"GPA vs. {feature}")
    plt.show()

    # (可选) 使用条形图显示每个类别的平均 GPA
    # mean_gpa = data.groupby(feature)['GPA'].mean()
    # mean_gpa.plot(kind='bar')
    # plt.ylabel('平均 GPA')
    # plt.title(f'平均 GPA vs. {feature}')
    # plt.show()

# 4.  绘制 GPA 的直方图

plt.figure(figsize=(8, 6))
plt.hist(
    data["GPA"], bins=20, color="skyblue", edgecolor="black"
)  # 可以调整 bins 的数量
plt.xlabel("GPA")
plt.ylabel("Frequency")
plt.title("GPA Distribution")
plt.show()

# 5.  (可选) 绘制 GradeClass 的直方图

plt.figure(figsize=(8, 6))
plt.hist(
    data["GradeClass"], bins=5, color="lightgreen", edgecolor="black"
)  # 可以调整 bins 的数量
plt.xlabel("GradeClass")
plt.ylabel("Frequency")
plt.title("GradeClass Distribution")
plt.show()

# import pandas as pd
# import numpy as np
# import scipy.stats as stats
# import matplotlib.pyplot as plt

# # 1. 读取数据
# data = pd.read_csv("Student_performance_data _.csv")

# # 2. 数据预处理 (可选)
# #   - 处理缺失值
# #   - 转换数据类型

# # 3. 相关系数分析

# # a. 皮尔逊相关系数 (线性相关)
# pearson_corr = data[["Age", "StudyTimeWeekly", "Absences", "GPA"]].corr(
#     method="pearson"
# )
# print("皮尔逊相关系数矩阵:\n", pearson_corr)

# # b. 斯皮尔曼秩相关系数 (非线性相关)
# spearman_corr = data[["Age", "StudyTimeWeekly", "Absences", "GPA"]].corr(
#     method="spearman"
# )
# print("\n斯皮尔曼秩相关系数矩阵:\n", spearman_corr)

# # 4. 假设检验

# # a. 卡方检验 (分类型变量相关性)
# #   - 假设：性别 (Gender) 与 GradeClass 是否相关？
# contingency_table = pd.crosstab(data["Gender"], data["GradeClass"])
# chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
# print("\n卡方检验结果:")
# print("卡方值:", chi2)
# print("p值:", p)
# print("自由度:", dof)
# print("预期频率表:\n", expected)

# alpha = 0.05  # 显著性水平
# if p < alpha:
#     print("拒绝原假设，性别与 GradeClass 之间存在显著相关性。")
# else:
#     print("接受原假设，性别与 GradeClass 之间不存在显著相关性。")

# # b. t检验 (组间差异分析)
# #   - 假设：不同性别 (Gender) 的学生的 GPA 是否存在显著差异？
# group1 = data[data["Gender"] == 0]["GPA"]  # 假设 0 代表一种性别
# group2 = data[data["Gender"] == 1]["GPA"]  # 假设 1 代表另一种性别
# t_statistic, p_value = stats.ttest_ind(group1, group2)
# print("\nt检验结果:")
# print("t统计量:", t_statistic)
# print("p值:", p_value)

# if p_value < alpha:
#     print("拒绝原假设，不同性别的学生的 GPA 存在显著差异。")
# else:
#     print("接受原假设，不同性别的学生的 GPA 不存在显著差异。")

# # 5. 蒙特卡洛模拟


# # 模拟 GPA 受 StudyTimeWeekly 和 ParentalSupport 影响
# def gpa_model(study_time, parental_support, noise_std=0.5):
#     """
#     模拟 GPA 的函数，受学习时间和父母支持的影响。
#     """
#     # 假设 GPA = 0.5 * StudyTimeWeekly + 0.3 * ParentalSupport + 噪声
#     gpa = 0.5 * study_time + 0.3 * parental_support + np.random.normal(0, noise_std)
#     return gpa


# # 设置模拟参数
# num_simulations = 1000
# study_time_values = data["StudyTimeWeekly"].values
# parental_support_values = data["ParentalSupport"].values

# # 存储模拟结果
# simulated_gpa = []

# # 进行蒙特卡洛模拟
# for i in range(num_simulations):
#     # 随机选择学习时间和父母支持的值
#     index = np.random.randint(0, len(study_time_values))
#     study_time = study_time_values[index]
#     parental_support = parental_support_values[index]

#     # 模拟 GPA
#     gpa = gpa_model(study_time, parental_support)
#     simulated_gpa.append(gpa)

# # 分析模拟结果
# simulated_gpa = np.array(simulated_gpa)
# print("\n蒙特卡洛模拟结果:")
# print("模拟 GPA 的平均值:", simulated_gpa.mean())
# print("模拟 GPA 的标准差:", simulated_gpa.std())

# # 绘制模拟 GPA 的直方图
# plt.figure(figsize=(8, 6))
# plt.hist(simulated_gpa, bins=30, color="lightcoral", edgecolor="black")
# plt.xlabel("Simulated GPA")
# plt.ylabel("Frequency")
# plt.title("Monte Carlo Simulation of GPA")
# plt.show()
