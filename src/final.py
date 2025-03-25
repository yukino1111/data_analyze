import pandas as pd
from scipy.stats import (
    chi2_contingency,
    ttest_ind,
    f_oneway,
    kruskal,
    mannwhitneyu,
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from tabulate import tabulate
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

font_path = "assets/fonts/PingFang-Medium.ttf"  # 替换为你系统中存在的中文字体文件路径
font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
categorical_features = [
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
]
numerical_features = ["Age", "StudyTimeWeekly", "Absences", "GPA"]


def descriptive_stats(data):

    # 定义离散型和非离散型变量

    # 创建一个空的 DataFrame 来存储结果
    variable_info = pd.DataFrame(columns=["变量名", "类型", "数据类型"])
    # 遍历所有列
    for col in data.columns:
        if col in categorical_features:
            var_type = "离散型"
        elif col in numerical_features:
            var_type = "连续型"
        else:
            var_type = "其他"
        data_type = str(data[col].dtype)
        # 将信息添加到 DataFrame
        variable_info = pd.concat(
            [
                variable_info,
                pd.DataFrame(
                    [{"变量名": col, "类型": var_type, "数据类型": data_type}]
                ),
            ],
            ignore_index=True,
        )
    # 打印 DataFrame (左对齐，无序号)
    print("数据集变量信息:")
    print(
        tabulate(
            variable_info,
            headers="keys",
            tablefmt="psql",
            showindex=False,
            colalign=("left", "left", "left"),
        )
    )

    # 统计离散型变量
    print("\n离散型变量统计:")
    for feature in categorical_features:
        print(f"{feature}:")  # 输出变量名
        counts = data[feature].value_counts()
        proportions = data[feature].value_counts(normalize=True)
        combined = pd.DataFrame({"count": counts, "proportion": proportions})
        combined.index.name = "value"  # 设置索引名为 "value"
        # 按照 value 值排序
        combined = combined.sort_index()
        # 使用字符串格式化对齐 "value" 列
        formatted_output = ""
        for index, row in combined.iterrows():
            formatted_output += (
                f"{str(index):<8} {int(row['count']):<8} {row['proportion']:.6f}\n"
            )
        print("值       数量     占比")
        print(formatted_output)

    # 统计非离散型变量
    print("非离散型变量统计:")
    descriptive_stats = (
        data[numerical_features].describe().loc[["mean", "std", "min", "max"]]
    )
    print(descriptive_stats)

    # 检查缺失值
    print("\n缺失值统计:")
    print(data.isnull().sum())

    # 检查重复值
    print(f"\n重复行的数量: \n{data.duplicated().sum()}")

    # 异常值检测 (考虑已知范围)
    print("\n异常值检测 (基于已知范围):")
    for feature in numerical_features + categorical_features:
        if feature == "Age":
            min_val, max_val = 15, 18
        elif feature == "StudyTimeWeekly":
            min_val, max_val = 0, 20
        elif feature == "Absences":
            min_val, max_val = 0, 30
        elif feature == "GPA":
            min_val, max_val = 0, 4.0  # Corrected GPA range
        elif feature == "Gender":
            min_val, max_val = 0, 1
        elif feature == "Ethnicity":
            min_val, max_val = 0, 3
        elif feature == "ParentalEducation":
            min_val, max_val = 0, 4
        elif feature == "Tutoring":
            min_val, max_val = 0, 1
        elif feature == "ParentalSupport":
            min_val, max_val = 0, 4
        elif feature == "Extracurricular":
            min_val, max_val = 0, 1
        elif feature == "Sports":
            min_val, max_val = 0, 1
        elif feature == "Music":
            min_val, max_val = 0, 1
        elif feature == "Volunteering":
            min_val, max_val = 0, 1
        else:
            min_val, max_val = (
                data[feature].min(),
                data[feature].max(),
            )  # 如果没有明确范围，则使用数据的最小值和最大值
        outliers = data[(data[feature] < min_val) | (data[feature] > max_val)][feature]
        if not outliers.empty:
            print(f"{feature} 中的异常值: \n{outliers}")
        else:
            print(f"{feature} 中没有发现超出范围的异常值")

    numeric_cols = data.columns
    num_cols = len(numeric_cols)
    n_rows = (num_cols + 2) // 3  # 向上取整，确保有足够的子图位置
    plt.figure(figsize=(15, n_rows * 4))  # 动态调整图表高度
    for i, col in enumerate(numeric_cols):
        plt.subplot(n_rows, 3, i + 1)  # 注意这里是i+1而不是i
        sns.boxplot(y=data[col])
        plt.title(f"{col}的箱线图")
    plt.tight_layout()
    plt.show()


def clean_wash(data):
    # 处理缺失值 (如果存在)
    # data = data.dropna()  # 删除包含缺失值的行
    # data = data.fillna(data.mean())  # 用平均值填充缺失值
    # 删除重复值
    # data = data.drop_duplicates()

    # 数据类型转换 (如果需要)
    # data['Age'] = data['Age'].astype(int)
    # 如果有缺失值，可以选择填充或删除
    # 例如，用平均值填充 GPA 列的缺失值：
    # data['GPA'].fillna(data['GPA'].mean(), inplace=True)

    # 将分类变量转换为数值型 (如果需要)
    # 例如，将 Gender 列转换为 0 和 1：
    # data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})  # 如果 Gender 列是字符串

    return data


# 异常值检测 (Z-score 法)
def detect_outliers_zscore(data, threshold=3):
    z = np.abs(stats.zscore(data))
    outlier_indices = np.where(z > threshold)
    return outlier_indices


def show_gpa(data):
    # GPA 分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data["GPA"], kde=True)
    plt.title("GPA分布", fontproperties=font)
    plt.xlabel("GPA", fontproperties=font)
    plt.ylabel("频率", fontproperties=font)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    plt.show()
    # 计算 GPA 的统计量
    gpa_mean = data["GPA"].mean()
    gpa_std = data["GPA"].std()
    gpa_ci = np.percentile(data["GPA"], [2.5, 97.5])
    print("\nGPA统计量：")
    print(f"  GPA均值: {gpa_mean:.3f}")
    print(f"  GPA标准差: {gpa_std:.3f}")
    print(f"  GPA的95%置信区间: [{gpa_ci[0]:.3f}, {gpa_ci[1]:.3f}]")


def calculate_correlations(data):

    # Calculate Pearson correlation
    pearson_corr = data.corr(method="pearson")
    # Calculate Spearman correlation
    spearman_corr = data.corr(method="spearman")
    # Create heatmaps
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title("皮尔逊相关系数热力图", fontproperties=font)
    plt.show()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        spearman_corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5
    )  # Changed fmt to .3f
    plt.title("斯皮尔曼相关系数热力图", fontproperties=font)
    plt.show()
    return pearson_corr, spearman_corr


def show_correlations(data):
    pearson_corr, spearman_corr = calculate_correlations(data)
    print("\n皮尔逊相关系数矩阵:")
    print(pearson_corr.to_string(float_format="{:.3f}".format))
    print("\n斯皮尔曼相关系数矩阵:")
    print(spearman_corr.to_string(float_format="{:.3f}".format))


def analyze_gpa_factors(data):
    # 定义 ParentalSupport 的映射
    parental_support_map = {0: "无", 1: "低", 2: "中", 3: "高", 4: "非常高"}
    # 创建图形
    plt.figure(figsize=(18, 6))
    # GPA vs. Absences (散点图)
    plt.subplot(1, 3, 1)
    sns.scatterplot(x="Absences", y="GPA", data=data)
    plt.xlabel("缺勤次数", fontproperties=font)
    plt.ylabel("绩点", fontproperties=font)
    plt.title("绩点vs缺勤次数", fontproperties=font)
    # GPA vs. StudyTimeWeekly (散点图)
    plt.subplot(1, 3, 2)
    sns.scatterplot(x="StudyTimeWeekly", y="GPA", data=data)
    plt.xlabel("每周学习时间", fontproperties=font)
    plt.ylabel("绩点", fontproperties=font)
    plt.title("绩点vs每周学习时间", fontproperties=font)
    # GPA vs. ParentalSupport (箱线图)
    plt.subplot(1, 3, 3)
    data_copy = data.copy()  # 创建副本，避免修改原始数据
    data_copy["ParentalSupport"] = data_copy["ParentalSupport"].map(
        parental_support_map
    )
    category_order = list(parental_support_map.values())  # 确保顺序
    ax = sns.boxplot(x="ParentalSupport", y="GPA", data=data_copy, order=category_order)
    plt.xlabel("家长支持程度", fontproperties=font)
    plt.ylabel("绩点", fontproperties=font)
    plt.title("绩点vs家长支持程度", fontproperties=font)
    # 设置 x 轴刻度和标签的字体
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    plt.tight_layout()  # 调整子图参数，以提供一个整齐的布局。
    plt.show()


def perform_chi2_test(data, variable):
    # 将 GPA 分类 (例如，高、中、低、优)
    data["GPA_Category"] = pd.cut(
        data["GPA"], bins=[0, 2, 3, 4], labels=["差", "中", "优"]
    )
    # 检查变量是分类变量还是连续变量
    if variable in numerical_features:
        # 如果是连续变量，则进行离散化
        data[variable + "_Category"] = pd.qcut(
            data[variable], q=3, labels=["低", "中", "高"]
        )
        variable_to_use = variable + "_Category"
        print(variable_to_use)
    else:
        variable_to_use = variable
    # 创建列联表
    contingency_table = pd.crosstab(data["GPA_Category"], data[variable_to_use])
    # 检查列联表是否至少有 2 行和 2 列
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        print(f"警告：{variable}的列联表的行数或列数少于 2。卡方检验可能无效。")
        return None
    # 可视化列联表 (热图)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        contingency_table,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "频数"},
    )
    # 设置标题和轴标签的字体
    plt.title(f"GPA和{variable}的列联表热图", fontproperties=font)
    plt.xlabel(variable, fontproperties=font)
    plt.ylabel("GPA类别", fontproperties=font)
    # 获取颜色条对象并设置标签字体
    cbar = ax.collections[0].colorbar
    cbar.set_label("频数", fontproperties=font)
    # 设置 x 轴和 y 轴刻度标签的字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    plt.show()

    # 执行卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"GPA和{variable}的卡方检验结果：")
    print(f"  卡方统计量: {chi2:.3f}")
    print(f"  P值: {p}")
    # print(f"  自由度: {dof}")
    # print(f"  期望频数:\n{expected}")  # 打印期望频数矩阵，可选
    alpha = 0.05  # 显著性水平
    if p < alpha:
        print(f"  结论：GPA和{variable}之间存在显著相关性。")
    else:
        print(f"  结论：GPA和{variable}之间不存在显著相关性。")
    return chi2, p, dof, expected


def perform_t_test(data, variable):
    # 根据中位数将数值变量分为两组
    median_value = data[variable].median()
    group1 = data[data[variable] <= median_value]["GPA"]
    group2 = data[data[variable] > median_value]["GPA"]
    # 检查两个组是否都有数据
    if len(group1) == 0 or len(group2) == 0:
        print(f"警告：{variable}在按中位数分割后，其中一个组为空。无法执行t检验。")
        return None
    # 执行 t 检验
    t_statistic, p_value = ttest_ind(group1, group2)
    print(f"GPA和{variable}的t检验结果：")
    print(f"  T 统计量: {t_statistic:.3f}")
    print(f"  P 值: {p_value}")
    alpha = 0.05  # 显著性水平
    if p_value < alpha:
        print(f"  结论：GPA在{variable}的两组之间存在显著差异。")
    else:
        print(f"  结论：GPA在{variable}的两组之间不存在显著差异。")
    return t_statistic, p_value


def perform_anova_test(data, variable, group_variable="GPA", num_bins=3):
    # 检查变量是否是数值类型 且 不是 ParentalSupport
    if pd.api.types.is_numeric_dtype(data[variable]) and variable != "ParentalSupport":
        # 如果是数值类型且不是 ParentalSupport，则进行分组
        temp_variable = variable + "_Category"  # 临时变量名
        data[temp_variable] = pd.cut(
            data[variable], bins=num_bins, labels=False
        )  # 使用数字标签
    else:
        temp_variable = variable  # 使用原始变量名
    # 获取不同组的数据
    groups = data[temp_variable].unique()
    group_data = [data[data[temp_variable] == g][group_variable] for g in groups]
    # 执行 ANOVA
    f_statistic, p_value = f_oneway(*group_data)

    print(f"{group_variable}和{variable}的ANOVA检验结果：")
    print(f"  F统计量: {f_statistic:.3f}")
    print(f"  P值: {p_value}")

    alpha = 0.05  # 显著性水平
    if p_value < alpha:
        print(f"  结论：{group_variable}在{variable}的不同水平之间存在显著差异。")
    else:
        print(f"  结论：{group_variable}在{variable}的不同水平之间不存在显著差异。")

    return f_statistic, p_value


def perform_kruskal_test(data, variable, group_variable="GPA"):
    # 获取不同组的数据
    groups = data[variable].unique()
    group_data = [data[data[variable] == g][group_variable] for g in groups]
    # 执行 Kruskal-Wallis H 检验
    h_statistic, p_value = kruskal(*group_data)

    print(f"{group_variable}和{variable}的Kruskal-Wallis H检验结果：")
    print(f"  H统计量: {h_statistic:.3f}")
    print(f"  P值: {p_value}")

    alpha = 0.05  # 显著性水平
    if p_value < alpha:
        print(f"  结论：{group_variable}在{variable}的不同水平的分布之间存在显著差异。")
    else:
        print(
            f"  结论：{group_variable}在{variable}的不同水平的分布之间不存在显著差异。"
        )

    return h_statistic, p_value


def perform_mannwhitneyu_test(data, variable):
    # 根据中位数将数值变量分为两组
    median_value = data[variable].median()
    group1 = data[data[variable] <= median_value]["GPA"]
    group2 = data[data[variable] > median_value]["GPA"]

    # 检查两个组是否都有数据
    if len(group1) == 0 or len(group2) == 0:
        print(
            f"警告：{variable}在按中位数分割后，其中一个组为空。无法执行Mann-Whitney U检验。"
        )
        return None

    # 执行 Mann-Whitney U 检验
    u_statistic, p_value = mannwhitneyu(group1, group2)

    print(f"GPA和{variable}的Mann-Whitney U检验结果：")
    print(f"  U统计量: {u_statistic:.3f}")
    print(f"  P值: {p_value:.3f}")

    alpha = 0.05  # 显著性水平
    if p_value < alpha:
        print(f"  结论：GPA在{variable}的两组之间的分布存在显著差异。")
    else:
        print(f"  结论：GPA在{variable}的两组之间的分布不存在显著差异。")

    return u_statistic, p_value


def show_chi2_test(data):
    print("\n卡方检验：")
    perform_chi2_test(data.copy(), "Absences")
    perform_chi2_test(data.copy(), "ParentalSupport")
    perform_chi2_test(data.copy(), "StudyTimeWeekly")


def show_t_test(data):
    print("\nt检验：")
    perform_t_test(data, "Absences")
    # perform_t_test(data, "ParentalSupport")
    perform_t_test(data, "StudyTimeWeekly")


def show_anova_test(data):
    print("\nANOVA检验：")
    perform_anova_test(data.copy(), "Absences")
    perform_anova_test(data.copy(), "ParentalSupport")
    perform_anova_test(data.copy(), "StudyTimeWeekly")


def show_kruskal_test(data):
    print("\nKruskal-Wallis H检验：")
    perform_kruskal_test(data.copy(), "Absences")
    perform_kruskal_test(data.copy(), "ParentalSupport")
    perform_kruskal_test(data.copy(), "StudyTimeWeekly")


def show_mannwhitneyu_test(data):
    print("\nMann-Whitney U检验：")
    perform_mannwhitneyu_test(data.copy(), "Absences")
    perform_mannwhitneyu_test(data.copy(), "ParentalSupport")
    perform_mannwhitneyu_test(data.copy(), "StudyTimeWeekly")


def show_standard_test():
    show_chi2_test(data)
    show_t_test(data)


def show_external_test():
    show_anova_test(data)
    # show_kruskal_test(data)
    # show_mannwhitneyu_test(data)


def smooth_clip(x, lower, upper, smoothness=1.0):
    """使用 sigmoid 函数平滑地将值限制在 lower 和 upper 之间."""
    return lower + (upper - lower) / (
        1 + np.exp(-smoothness * (x - (lower + upper) / 2))
    )


def perform_linear_regression_mc(data, n_simulations=2400, use_all_variables=True):
    # 1. 定义变量的概率分布
    absences_mean = data["Absences"].mean()
    absences_std = data["Absences"].std()

    if use_all_variables:
        studytime_mean = data["StudyTimeWeekly"].mean()
        studytime_std = data["StudyTimeWeekly"].std()
        parental_support_probs = data["ParentalSupport"].value_counts(normalize=True)
        variables = ["Absences", "StudyTimeWeekly", "ParentalSupport"]
        title_suffix = " (使用Absences, StudyTimeWeekly, ParentalSupport)"
    else:
        variables = ["Absences"]
        title_suffix = " (仅使用Absences)"

    # 2. 构建 GPA 的预测模型 (线性回归)
    model = LinearRegression()
    X = data[variables]
    y = data["GPA"]
    model.fit(X, y)

    # 3. 执行蒙特卡洛模拟
    simulated_gpas = []
    for _ in range(n_simulations):
        absences_sample = np.random.normal(absences_mean, absences_std)

        if use_all_variables:
            studytime_sample = np.random.normal(studytime_mean, studytime_std)
            parental_support_sample = np.random.choice(
                parental_support_probs.index, p=parental_support_probs.values
            )
            input_df = pd.DataFrame(
                [[absences_sample, studytime_sample, parental_support_sample]],
                columns=variables,
            )
        else:
            input_df = pd.DataFrame([[absences_sample]], columns=variables)

        gpa_prediction = model.predict(input_df)[0]
        # gpa_prediction = np.clip(gpa_prediction, 0, 4.0)  # 截断
        gpa_prediction = smooth_clip(gpa_prediction, 0, 4.0)
        simulated_gpas.append(gpa_prediction)

    # 4. 分析模拟结果
    simulated_gpas = np.array(simulated_gpas)

    print(f"线性回归蒙特卡洛模拟结果 (模拟次数: {n_simulations}){title_suffix}:")
    print(f"  模拟GPA均值: {simulated_gpas.mean():.3f}")
    print(f"  模拟GPA标准差: {simulated_gpas.std():.3f}")
    s_gpa_ci = np.percentile(simulated_gpas, [2.5, 97.5])
    print(f"  模拟GPA的95%置信区间: [{s_gpa_ci[0]:.3f}, {s_gpa_ci[1]:.3f}]")

    plt.figure(figsize=(10, 6))
    sns.histplot(simulated_gpas, kde=True)
    plt.title(f"模拟的GPA分布 - 线性回归{title_suffix}", fontproperties=font)
    plt.xlabel("GPA", fontproperties=font)
    plt.ylabel("频率", fontproperties=font)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    plt.show()

    return simulated_gpas


def perform_polynomial_regression_mc(
    data, n_simulations=20000, degree=2, use_all_variables=True
):
    # 1. 定义变量的概率分布
    absences_mean = data["Absences"].mean()
    absences_std = data["Absences"].std()

    if use_all_variables:
        studytime_mean = data["StudyTimeWeekly"].mean()
        studytime_std = data["StudyTimeWeekly"].std()
        parental_support_probs = data["ParentalSupport"].value_counts(normalize=True)
        variables = ["Absences", "StudyTimeWeekly", "ParentalSupport"]
        title_suffix = " (使用Absences, StudyTimeWeekly, ParentalSupport)"
    else:
        variables = ["Absences"]
        title_suffix = " (仅使用Absences)"

    # 2. 构建 GPA 的预测模型 (多项式回归)
    poly = PolynomialFeatures(degree=degree)
    X = data[variables]
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, data["GPA"])

    # 3. 执行蒙特卡洛模拟
    simulated_gpas = []
    for _ in range(n_simulations):
        absences_sample = np.random.normal(absences_mean, absences_std)

        if use_all_variables:
            studytime_sample = np.random.normal(studytime_mean, studytime_std)
            parental_support_sample = np.random.choice(
                parental_support_probs.index, p=parental_support_probs.values
            )
            input_df = pd.DataFrame(
                [[absences_sample, studytime_sample, parental_support_sample]],
                columns=variables,
            )
        else:
            input_df = pd.DataFrame([[absences_sample]], columns=variables)

        input_poly = poly.transform(input_df)  # 转换
        gpa_prediction = model.predict(input_poly)[0]
        # gpa_prediction = np.clip(gpa_prediction, 0, 4.0)  # 截断
        gpa_prediction = smooth_clip(gpa_prediction, 0, 4.0)
        simulated_gpas.append(gpa_prediction)

    # 4. 分析模拟结果
    simulated_gpas = np.array(simulated_gpas)

    print(
        f"多项式回归蒙特卡洛模拟结果 (模拟次数: {n_simulations}, 阶数: {degree}){title_suffix}:"
    )
    print(f"  模拟GPA均值: {simulated_gpas.mean():.3f}")
    print(f"  模拟GPA标准差: {simulated_gpas.std():.3f}")
    s_gpa_ci = np.percentile(simulated_gpas, [2.5, 97.5])
    print(f"  模拟GPA的95%置信区间: [{s_gpa_ci[0]:.3f}, {s_gpa_ci[1]:.3f}]")

    plt.figure(figsize=(10, 6))
    sns.histplot(simulated_gpas, kde=True)
    plt.title(
        f"模拟的GPA分布 - 多项式回归 (阶数: {degree}){title_suffix}",
        fontproperties=font,
    )
    plt.xlabel("GPA", fontproperties=font)
    plt.ylabel("频率", fontproperties=font)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    plt.show()

    return simulated_gpas


def show_regression_mc(data):
    print("\n蒙特卡洛模拟：")
    # 线性回归 (仅使用 Absences)
    perform_linear_regression_mc(data.copy(), use_all_variables=False)
    # 线性回归 (使用Absences, StudyTimeWeekly, ParentalSupport)
    perform_linear_regression_mc(data.copy(), use_all_variables=True)
    # 多项式回归 (仅使用 Absences, 阶数=2)
    # perform_polynomial_regression_mc(data.copy(), degree=2, use_all_variables=False)
    # 多项式回归 (使用Absences, StudyTimeWeekly, ParentalSupport, 阶数=2)
    # perform_polynomial_regression_mc(data.copy(), degree=2, use_all_variables=True)


if __name__ == "__main__":
    data = pd.read_csv("./assets/Student_performance_data _.csv")
    data = data.drop("StudentID", axis=1)
    data = data.drop("GradeClass", axis=1)
    # data=clean_wash(data)
    # descriptive_stats(data)
    # show_gpa(data)
    # show_correlations(data)
    # analyze_gpa_factors(data)
    # show_standard_test()
    # show_external_test()
    show_regression_mc(data)
