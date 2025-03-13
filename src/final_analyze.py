import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


# 1. 读取数据
# 设置中文字体
font_path = "assets/fonts/PingFang-Medium.ttf"  # 替换为你系统中存在的中文字体文件路径
font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

data = pd.read_csv("./assets/Student_performance_data _.csv")

# 2. 数据预处理 (略，与图表修改无关)

# 3. 分析 GPA 与其他因素的关系

numerical_features = [
    "Age",
    "StudyTimeWeekly",
    "Absences",
]
numerical_features_cn = {
    "Age": "年龄",
    "StudyTimeWeekly": "每周学习时间",
    "Absences": "缺勤次数",
}
parental_support_map = {0: "无", 1: "低", 2: "中等", 3: "高", 4: "非常高"}
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(data[feature], data["GPA"])
    plt.xlabel(numerical_features_cn[feature], fontproperties=font)
    plt.ylabel("绩点", fontproperties=font)  # GPA 改为 绩点
    plt.title(f"绩点与 {numerical_features_cn[feature]} 的关系", fontproperties=font)

    if feature == "Age":
        plt.xticks(range(int(data["Age"].min()), int(data["Age"].max()) + 1))  # 去掉 .5

    plt.show()

    correlation = data[feature].corr(data["GPA"])
    print(f"绩点与 {numerical_features_cn[feature]} 的相关系数: {correlation}")

categorical_features = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "ParentalSupport",
    "Tutoring",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]

categorical_features_cn = {
    "Gender": "性别",
    "Ethnicity": "种族",
    "ParentalEducation": "父母教育程度",
    "ParentalSupport": "父母支持程度",
    "Tutoring": "是否辅导",
    "Extracurricular": "是否参加课外活动",
    "Sports": "是否参加体育活动",
    "Music": "是否参加音乐活动",
    "Volunteering": "是否参加志愿活动",
}

# 数据字典翻译
gender_map = {0: "男", 1: "女"}
ethnicity_map = {0: "高加索人", 1: "非裔美国人", 2: "亚裔", 3: "其他"}
parental_education_map = {0: "无", 1: "高中", 2: "大学专科", 3: "学士", 4: "更高"}
binary_map = {0: "否", 1: "是"}  # 用于二元分类变量


for feature in categorical_features:
    plt.figure(figsize=(8, 6))

    # 根据不同的特征使用不同的映射
    if feature == "Gender":
        data_copy = data.copy()  # 创建副本，避免修改原始数据
        data_copy[feature] = data_copy[feature].map(gender_map)
        category_order = list(gender_map.values())  # 确保顺序
        ax = sns.boxplot(
            x=feature, y="GPA", data=data_copy, order=category_order
        )  # 保存boxplot的axes
    elif feature == "Ethnicity":
        data_copy = data.copy()
        data_copy[feature] = data_copy[feature].map(ethnicity_map)
        category_order = list(ethnicity_map.values())  # 确保顺序
        ax = sns.boxplot(
            x=feature, y="GPA", data=data_copy, order=category_order
        )  # 保存boxplot的axes
    elif feature == "ParentalEducation":
        data_copy = data.copy()
        data_copy[feature] = data_copy[feature].map(parental_education_map)
        category_order = list(parental_education_map.values())  # 确保顺序
        ax = sns.boxplot(
            x=feature, y="GPA", data=data_copy, order=category_order
        )  # 保存boxplot的axes
    elif feature == "ParentalSupport":  # 添加 ParentalSupport 的处理
        data_copy = data.copy()
        data_copy[feature] = data_copy[feature].map(parental_support_map)
        category_order = list(parental_support_map.values())  # 确保顺序
        ax = sns.boxplot(
            x=feature, y="GPA", data=data_copy, order=category_order
        )  # 保存boxplot的axes
    elif feature in ["Tutoring", "Extracurricular", "Sports", "Music", "Volunteering"]:
        data_copy = data.copy()
        data_copy[feature] = data_copy[feature].map(binary_map)
        category_order = list(binary_map.values())  # 确保顺序
        ax = sns.boxplot(
            x=feature, y="GPA", data=data_copy, order=category_order
        )  # 保存boxplot的axes
    else:
        ax = sns.boxplot(x=feature, y="GPA", data=data)  # 其他情况 # 保存boxplot的axes

    plt.xlabel(categorical_features_cn[feature], fontproperties=font)
    plt.ylabel("绩点", fontproperties=font)
    plt.title(f"绩点与 {categorical_features_cn[feature]} 的关系", fontproperties=font)

    # 获取当前 axes 对象
    ax = plt.gca()  # 或者使用之前保存的 ax

    # 循环遍历 x 轴上的每个标签，并设置字体属性
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)

    # 设置 x 轴刻度和标签
    if feature in [
        "Gender",
        "Ethnicity",
        "ParentalEducation",
        "ParentalSupport",
        "Tutoring",
        "Extracurricular",
        "Sports",
        "Music",
        "Volunteering",
    ]:
        ax.set_xticks(range(len(category_order)))  # 设置刻度位置
        ax.set_xticklabels(category_order)  # 设置刻度标签，确保顺序

        # 再次循环设置字体 (因为 set_xticklabels 可能会重置字体)
        for label in ax.get_xticklabels():
            label.set_fontproperties(font)

    plt.show()


# 4. 绘制 GPA 的直方图
plt.figure(figsize=(8, 6))
plt.hist(data["GPA"], bins=20, color="skyblue", edgecolor="black")
plt.xlabel("绩点", fontproperties=font)
plt.ylabel("频数", fontproperties=font)
plt.title("绩点分布", fontproperties=font)
plt.show()

# 5. (可选) 绘制 GradeClass 的直方图
plt.figure(figsize=(8, 6))
plt.hist(data["GradeClass"], bins=5, color="lightgreen", edgecolor="black")
plt.xlabel("成绩等级", fontproperties=font)
plt.ylabel("频数", fontproperties=font)
plt.title("成绩等级分布", fontproperties=font)
plt.show()

# # 3. 相关系数分析

# # a. 皮尔逊相关系数 (线性相关)
# 定义中文列名映射
column_names_cn = {
    "Age": "年龄",
    "StudyTimeWeekly": "每周学习时间",
    "Absences": "缺勤次数",
    "GPA": "绩点",
}

pearson_corr = data[["Age", "StudyTimeWeekly", "Absences", "GPA"]].corr(
    method="pearson"
)
# 重命名列名
pearson_corr = pearson_corr.rename(columns=column_names_cn, index=column_names_cn)
print("皮尔逊相关系数矩阵:\n", pearson_corr)

# # b. 斯皮尔曼秩相关系数 (非线性相关)
spearman_corr = data[["Age", "StudyTimeWeekly", "Absences", "GPA"]].corr(
    method="spearman"
)
# 重命名列名
spearman_corr = spearman_corr.rename(columns=column_names_cn, index=column_names_cn)
print("\n斯皮尔曼秩相关系数矩阵:\n", spearman_corr)


# # 4. 假设检验

# # a. 卡方检验 (分类型变量相关性)
# #   - 假设：性别 (Gender) 与 GradeClass 是否相关？
contingency_table = pd.crosstab(data["Gender"], data["GradeClass"])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\n卡方检验结果:")
print("卡方值:", chi2)
print("p值:", p)
print("自由度:", dof)
print("预期频率表:\n", expected)

alpha = 0.05  # 显著性水平
if p < alpha:
    print("拒绝原假设，性别与 成绩等级 之间存在显著相关性。")
else:
    print("接受原假设，性别与 成绩等级 之间不存在显著相关性。")

# # b. t检验 (组间差异分析)
# #   - 假设：不同性别 (Gender) 的学生的 GPA 是否存在显著差异？
group1 = data[data["Gender"] == 0]["GPA"]  # 假设 0 代表一种性别
group2 = data[data["Gender"] == 1]["GPA"]  # 假设 1 代表另一种性别
t_statistic, p_value = stats.ttest_ind(group1, group2)
print("\nt检验结果:")
print("t统计量:", t_statistic)
print("p值:", p_value)

if p_value < alpha:
    print("拒绝原假设，不同性别的学生的 绩点 存在显著差异。")
else:
    print("接受原假设，不同性别的学生的 绩点 不存在显著差异。")

# # 5. 蒙特卡洛模拟


# 2. 准备训练数据
X = data[["ParentalSupport", "Absences"]]  # 特征
y = data["GPA"]  # 目标变量
# 3. 训练线性回归模型
model = LinearRegression()
model.fit(X, y)
# 4. 获取训练好的模型参数
base_gpa = model.intercept_  # 截距
parental_support_effect = model.coef_[0]  # ParentalSupport 的系数
absences_effect = model.coef_[1]  # Absences 的系数
print("训练好的模型参数:")
print("基础 GPA:", base_gpa)
print("父母支持影响:", parental_support_effect)
print("缺勤次数影响:", absences_effect)


# 定义 gpa_model 函数 (使用训练好的参数)
def gpa_model(parental_support, absences):
    """
    模拟 GPA 的函数 (使用训练好的参数)。
    Args:
        parental_support: 父母支持程度 (0-4)。
        absences: 缺勤次数。
    Returns:
        模拟的 GPA 值 (0-4.0)。
    """
    # 计算 GPA
    gpa = (
        base_gpa
        + parental_support_effect * parental_support
        + absences_effect * absences
    )
    # 确保 GPA 在 0-4.0 范围内
    gpa = np.clip(gpa, 0, 4.0)
    return gpa


# 设置模拟参数
num_simulations = 1000
# 5. 根据源文件分布生成模拟数据
# 5.1 计算 ParentalSupport 的频率分布
parental_support_counts = data["ParentalSupport"].value_counts(normalize=True)
parental_support_values = parental_support_counts.index.to_numpy()
parental_support_probabilities = parental_support_counts.to_numpy()
# 5.2 计算 Absences 的频率分布
absences_counts = data["Absences"].value_counts(normalize=True)
absences_values = absences_counts.index.to_numpy()
absences_probabilities = absences_counts.to_numpy()
# 5.3 使用计算出的概率进行抽样
simulated_parental_support = np.random.choice(
    parental_support_values, size=num_simulations, p=parental_support_probabilities
)
simulated_absences = np.random.choice(
    absences_values, size=num_simulations, p=absences_probabilities
)
# 存储模拟结果
simulated_gpa = []
# 设置随机数种子，以确保结果可重复
np.random.seed(42)
# 进行蒙特卡洛模拟
for i in range(num_simulations):
    # 使用随机生成的 parental_support 和 absences 值
    parental_support = simulated_parental_support[i]
    absences = simulated_absences[i]
    # 模拟 GPA
    gpa = gpa_model(parental_support, absences)
    simulated_gpa.append(gpa)
# 分析模拟结果
simulated_gpa = np.array(simulated_gpa)
print("\n蒙特卡洛模拟结果:")
print("模拟 绩点 的平均值:", simulated_gpa.mean())
print("模拟 绩点 的标准差:", simulated_gpa.std())
# 绘制频率直方图和散点图
plt.figure(figsize=(20, 10))  # 调整整体图形大小
# 1. 父母支持程度对 GPA 的影响 (散点图)
plt.subplot(2, 3, 1)  # 2 行 3 列，第一个子图
plt.scatter(simulated_parental_support, simulated_gpa, color="skyblue")
plt.xlabel("模拟父母支持程度", fontproperties=font)
plt.ylabel("模拟绩点", fontproperties=font)
plt.title("父母支持程度对 绩点 的影响", fontproperties=font)
plt.xticks(sorted(np.unique(simulated_parental_support)))  # 确保 x 轴刻度是整数
# 2. 缺勤次数对 GPA 的影响 (散点图)
plt.subplot(2, 3, 2)  # 2 行 3 列，第二个子图
plt.scatter(simulated_absences, simulated_gpa, color="lightgreen")
plt.xlabel("模拟缺勤次数", fontproperties=font)
plt.ylabel("模拟绩点", fontproperties=font)
plt.title("缺勤次数对 绩点 的影响", fontproperties=font)
# 3. 模拟的父母支持程度 (频率直方图)
plt.subplot(2, 3, 4)  # 2 行 3 列，第四个子图
plt.hist(
    simulated_parental_support,
    bins=len(np.unique(simulated_parental_support)),
    color="skyblue",
    edgecolor="black",
)
plt.xlabel("模拟父母支持程度", fontproperties=font)
plt.ylabel("频数", fontproperties=font)
plt.title("模拟父母支持程度的频率分布", fontproperties=font)
plt.xticks(sorted(np.unique(simulated_parental_support)))  # 确保 x 轴刻度是整数
# 4. 模拟的缺勤次数 (频率直方图)
plt.subplot(2, 3, 5)  # 2 行 3 列，第五个子图
plt.hist(
    simulated_absences, bins=30, color="lightgreen", edgecolor="black"
)  # bins 可以根据实际情况调整
plt.xlabel("模拟缺勤次数", fontproperties=font)
plt.ylabel("频数", fontproperties=font)
plt.title("模拟缺勤次数的频率分布", fontproperties=font)
# 5. 模拟的 GPA (频率直方图)
plt.subplot(2, 3, 6)  # 2 行 3 列，第六个子图
plt.hist(simulated_gpa, bins=30, color="lightcoral", edgecolor="black")
plt.xlabel("模拟绩点", fontproperties=font)
plt.ylabel("频数", fontproperties=font)
plt.title("模拟绩点的频率分布", fontproperties=font)
plt.xlim(0, 4.0)  # 设置 x 轴范围为 0-4.0
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
