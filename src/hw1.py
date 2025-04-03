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
import numpy as np
from sklearn.linear_model import LinearRegression
from config import *
from common import description_wash


def calculate_correlations(data):

    # Calculate Pearson correlation
    pearson_corr = data.corr(method="pearson")
    # Calculate Spearman correlation
    spearman_corr = data.corr(method="spearman")
    # Create heatmaps
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title("皮尔逊相关系数热力图", fontproperties=font)
    plt.savefig("./assets/pic/皮尔逊相关系数热力图.png")
    plt.show()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        spearman_corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5
    )  # Changed fmt to .3f
    plt.title("斯皮尔曼相关系数热力图", fontproperties=font)
    plt.savefig("./assets/pic/斯皮尔曼相关系数热力图.png")
    plt.show()
    return pearson_corr, spearman_corr


def show_correlations(data):
    pearson_corr, spearman_corr = calculate_correlations(data)
    print("\n皮尔逊相关系数矩阵:")
    print(pearson_corr.to_string(float_format="{:.3f}".format))
    print("\n斯皮尔曼相关系数矩阵:")
    print(spearman_corr.to_string(float_format="{:.3f}".format))


def analyze_gpa_factors(data):
    # 定义 ParentalInvolvement 的映射
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
    # GPA vs. ParentalInvolvement (箱线图)
    plt.subplot(1, 3, 3)
    data_copy = data.copy()  # 创建副本，避免修改原始数据
    data_copy["ParentalInvolvement"] = data_copy["ParentalInvolvement"].map(
        parental_support_map
    )
    category_order = list(parental_support_map.values())  # 确保顺序
    ax = sns.boxplot(
        x="ParentalInvolvement", y="GPA", data=data_copy, order=category_order
    )
    plt.xlabel("家长支持程度", fontproperties=font)
    plt.ylabel("绩点", fontproperties=font)
    plt.title("绩点vs家长支持程度", fontproperties=font)
    # 设置 x 轴刻度和标签的字体
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(font)
    plt.tight_layout()  # 调整子图参数，以提供一个整齐的布局。
    plt.savefig("./assets/pic/主要因素.png")
    plt.show()


def perform_chi2_test(data, variable):
    # 将 GPA 分类 (例如，高、中、低、优)
    data["GPA_Category"] = pd.cut(
        data["GPA"], bins=[0, 1, 2.5, 4], labels=["差", "中", "优"]
    )
    # 检查变量是分类变量还是连续变量
    if variable in numerical_features:
        # 如果是连续变量，则进行离散化
        data[variable + "_Category"] = pd.qcut(
            data[variable], q=3, labels=["低", "中", "高"]
        )
        variable_to_use = variable + "_Category"
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
    plt.savefig("./assets/pic/联列表.png")
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
    # 检查变量是否是数值类型 且 不是 ParentalInvolvement
    if (
        pd.api.types.is_numeric_dtype(data[variable])
        and variable != "ParentalInvolvement"
    ):
        # 如果是数值类型且不是 ParentalInvolvement，则进行分组
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
    perform_chi2_test(data.copy(), "ParentalInvolvement")
    perform_chi2_test(data.copy(), "StudyTimeWeekly")


def show_t_test(data):
    print("\nt检验：")
    perform_t_test(data, "Absences")
    # perform_t_test(data, "ParentalInvolvement")
    perform_t_test(data, "StudyTimeWeekly")


def show_anova_test(data):
    print("\nANOVA检验：")
    perform_anova_test(data.copy(), "Absences")
    # perform_anova_test(data.copy(), "ParentalInvolvement")
    perform_anova_test(data.copy(), "StudyTimeWeekly")


def show_kruskal_test(data):
    print("\nKruskal-Wallis H检验：")
    perform_kruskal_test(data.copy(), "Absences")
    perform_kruskal_test(data.copy(), "ParentalInvolvement")
    perform_kruskal_test(data.copy(), "StudyTimeWeekly")


def show_mannwhitneyu_test(data):
    print("\nMann-Whitney U检验：")
    perform_mannwhitneyu_test(data.copy(), "Absences")
    perform_mannwhitneyu_test(data.copy(), "ParentalInvolvement")
    perform_mannwhitneyu_test(data.copy(), "StudyTimeWeekly")


def show_standard_test():
    show_chi2_test(data)
    show_t_test(data)


def show_external_test():
    show_anova_test(data)
    # show_kruskal_test(data)
    # show_mannwhitneyu_test(data)


def smooth_clip(x, lower, upper, smoothness=1.0):
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
        parental_support_probs = data["ParentalInvolvement"].value_counts(
            normalize=True
        )
        variables = ["Absences", "StudyTimeWeekly", "ParentalInvolvement"]
        title_suffix = " (使用Absences, StudyTimeWeekly, ParentalInvolvement)"
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
    sns.histplot(simulated_gpas)
    plt.title(f"模拟的GPA分布 - 线性回归{title_suffix}", fontproperties=font)
    plt.xlabel("GPA", fontproperties=font)
    plt.ylabel("频率", fontproperties=font)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    plt.savefig("./assets/pic/线性回归.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    # 模拟数据的 GPA KDE 曲线
    sns.kdeplot(simulated_gpas, color="red", label="模拟GPA", linewidth=2)

    # 设置图表标题和标签
    plt.title("模拟GPA分布")
    plt.xlabel("GPA")
    plt.ylabel("密度")
    plt.xlim(0, 4)
    # 显示图例
    plt.legend()

    # 显示图形
    plt.savefig("./assets/pic/模拟gpa线.png")
    plt.show()

    return simulated_gpas


def show_regression_mc(data):
    print("\n蒙特卡洛模拟：")
    # 线性回归 (仅使用 Absences)
    perform_linear_regression_mc(data.copy(), use_all_variables=False)
    # 线性回归 (使用Absences, StudyTimeWeekly, ParentalInvolvement)
    perform_linear_regression_mc(data.copy(), use_all_variables=True)


if __name__ == "__main__":
    data = description_wash()
    
    show_correlations(data)
    analyze_gpa_factors(data)
    show_standard_test()
    show_external_test()
    show_regression_mc(data)
