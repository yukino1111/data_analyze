import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind

# 1. 读取数据
data = pd.read_csv("Student_performance_data _.csv")

# 2. 数据预处理
# 检查缺失值
print("缺失值统计:")
print(data.isnull().sum())

# 处理缺失值 (如果存在)
# data = data.dropna()  # 删除包含缺失值的行
# data = data.fillna(data.mean())  # 用平均值填充缺失值

# 检查重复值
print(f"重复行的数量: {data.duplicated().sum()}")

# 删除重复值
data = data.drop_duplicates()

# 数据类型转换 (如果需要)
# data['Age'] = data['Age'].astype(int)


# 3. 异常值检测 (Z-score 法)
def detect_outliers_zscore(data, threshold=3):
    z = np.abs(stats.zscore(data))
    outlier_indices = np.where(z > threshold)
    return outlier_indices


numerical_features = ["Age", "StudyTimeWeekly", "Absences", "ParentalSupport"]
for feature in numerical_features:
    outlier_indices = detect_outliers_zscore(data[feature])
    print(f"'{feature}' 列的异常值索引: {outlier_indices}")
    # 可以选择删除或替换这些异常值
    # data = data.drop(outlier_indices[0])

# 4. 相关性分析
numerical_features = ["Age", "StudyTimeWeekly", "Absences", "ParentalSupport"]

# 皮尔逊相关系数
print("\n皮尔逊相关系数:")
for feature in numerical_features:
    correlation = data[feature].corr(data["GPA"])
    print(f"{feature} 与 GPA 的相关系数: {correlation}")

# 斯皮尔曼秩相关系数
print("\n斯皮尔曼秩相关系数:")
for feature in numerical_features:
    correlation_spearman = data[feature].corr(data["GPA"], method="spearman")
    print(f"{feature} 与 GPA 的斯皮尔曼秩相关系数: {correlation_spearman}")

# 5. 假设检验 (卡方检验)
categorical_features = ["Gender", "Ethnicity", "ParentalEducation", "Tutoring"]
print("\n卡方检验:")
for feature in categorical_features:
    contingency_table = pd.crosstab(data[feature], data["GradeClass"])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"{feature} 与 GradeClass 的卡方检验:")
    print(f"  卡方值: {chi2}")
    print(f"  P 值: {p}")
    # print("  期望频率:")
    # print(expected)

# 6. 假设检验 (t 检验)
print("\nt 检验:")
group1 = data[data["Tutoring"] == 1]["GPA"]  # 接受辅导的学生
group2 = data[data["Tutoring"] == 0]["GPA"]  # 未接受辅导的学生
t_statistic, p_value = ttest_ind(group1, group2)
print(f"接受辅导与未接受辅导的学生 GPA 的 t 检验:")
print(f"  T 统计量: {t_statistic}")
print(f"  P 值: {p_value}")

# 7. 可视化
# 相关性热力图
correlation_matrix = data[numerical_features + ["GPA"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("相关性热力图")
plt.show()

# 散点图矩阵
sns.pairplot(data[numerical_features + ["GPA"]])
plt.suptitle("散点图矩阵", y=1.02)  # 调整标题位置
plt.show()

# 箱线图
categorical_features = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=feature, y="GPA", data=data)
    plt.xlabel(feature)
    plt.ylabel("GPA")
    plt.title(f"GPA vs. {feature}")
    plt.show()


# 蒙特卡洛模拟
def gpa_model(study_time, parental_support, noise_std=0.1):
    """一个简单的 GPA 模型，带有随机噪声。"""
    # 假设 GPA = 0.1 * StudyTimeWeekly + 0.2 * ParentalSupport + 噪声
    gpa = 0.1 * study_time + 0.2 * parental_support + np.random.normal(0, noise_std)
    return gpa


# 蒙特卡洛模拟
def monte_carlo_simulation(num_simulations=1000):
    """运行蒙特卡洛模拟来估计 GPA 的分布。"""
    gpa_values = []
    for _ in range(num_simulations):
        # 从数据集中随机抽样 StudyTimeWeekly 和 ParentalSupport
        sample = data.sample(n=1, replace=True)
        study_time = sample["StudyTimeWeekly"].values[0]
        parental_support = sample["ParentalSupport"].values[0]

        # 计算 GPA
        gpa = gpa_model(study_time, parental_support)
        gpa_values.append(gpa)

    return gpa_values


# 运行模拟
num_simulations = 1000
gpa_values = monte_carlo_simulation(num_simulations)

# 将结果转换为 Pandas Series 以便分析
gpa_series = pd.Series(gpa_values)

# 打印统计信息
print("蒙特卡洛模拟结果:")
print(gpa_series.describe())

# 可视化结果
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(gpa_series, kde=True)
plt.title("蒙特卡洛模拟 GPA 分布")
plt.xlabel("GPA")
plt.ylabel("频率")
plt.show()

# 8. 分析结论 (根据你的结果进行总结)
print("\n分析结论:")
print("  (根据你的分析结果进行总结，例如：)")
print("  - 学习时间与 GPA 呈正相关。")
print("  - 父母教育程度与 GPA 呈正相关。")
print("  - 接受辅导的学生的 GPA 显著高于未接受辅导的学生。")
print("  - ...")
print("\n方法的局限性:")
print("  - 相关系数只能衡量线性关系。")
print("  - 假设检验只能说明关联性，不能说明因果关系。")
print("  - 蒙特卡洛模拟的结果取决于模型的假设。")
