import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from config import *


def descriptive_stats(data):

    # 定义离散型和非离散型变量

    # 创建一个空的 DataFrame 来存储结果
    variable_info = pd.DataFrame(columns=["变量名", "类型"])
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
                pd.DataFrame([{"变量名": col, "类型": var_type}]),
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
            colalign=("left", "left"),
        )
    )

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
        elif feature == "ParentalInvolvement":
            min_val, max_val = 0, 4
        elif feature == "Extracurricular":
            min_val, max_val = 0, 1
        elif feature == "Sports":
            min_val, max_val = 0, 1
        elif feature == "Music":
            min_val, max_val = 0, 1
        elif feature == "Volunteering":
            min_val, max_val = 0, 1
        elif feature == "GradeClass":
            min_val, max_val = 0, 4
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
    plt.savefig("./assets/pic/箱型图.png")
    plt.show()


def describe_stats(data):
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

    # 绘制分布图 (n x n 形式)
    num_cols = len(data.columns)
    n = int(np.ceil(np.sqrt(num_cols)))  # 计算 n，用于 n x n 的子图排列
    plt.figure(figsize=(11, 11))  # 调整整体图形大小
    for i, column in enumerate(data.columns):
        plt.subplot(n, n, i + 1)  # 创建子图 (n x n 排列)
        sns.histplot(data[column])  # 绘制直方图和KDE曲线
        plt.title(f"{column} 的分布")  # 设置标题
        plt.xlabel(column)  # 设置x轴标签
        plt.ylabel("频率")  # 设置y轴标签
        # 定制化 x 轴刻度
        if column in [
            "Gender",
            "Tutoring",
            "Extracurricular",
            "Sports",
            "Music",
            "Volunteering",
        ]:
            plt.xticks([0, 1])  # 设置 x 轴刻度为 0 和 1
        elif column == "ParentalEducation":
            plt.xticks(range(5))  # 设置 x 轴刻度为 0, 1, 2, 3, 4
    plt.tight_layout()  # 自动调整子图参数，避免重叠
    plt.savefig("./assets/pic/分布.png")
    plt.show()


def clean_wash(data):
    # 处理缺失值 (如果存在)
    if data["Age"].isnull().any():
        mean_age = data["Age"].mean()
        data["Age"] = data["Age"].fillna(mean_age).round().astype(int)
    cols_to_drop = [col for col in data.columns if col not in ["Age"]]
    # 遍历要删除的列，删除包含缺失值的行
    for col in cols_to_drop:
        data.dropna(subset=[col], inplace=True)

    # 打印处理后的缺失值统计
    # print("\n缺失值统计:")
    # print(data.isnull().sum())

    feature_ranges = {
        "Age": (15, 18),
        "StudyTimeWeekly": (0, 20),
        "Absences": (0, 30),
        "GPA": (0, 4.0),
        "Gender": (0, 1),
        "Ethnicity": (0, 3),
        "ParentalEducation": (0, 4),
        "Tutoring": (0, 1),
        "ParentalInvolvement": (0, 4),
        "Extracurricular": (0, 1),
        "Sports": (0, 1),
        "Music": (0, 1),
        "Volunteering": (0, 1),
        "GradeClass": (0, 4),
    }

    for feature in feature_ranges:
        min_val, max_val = feature_ranges[feature]
        outliers = data[(data[feature] < min_val) | (data[feature] > max_val)][feature]
        if not outliers.empty:
            if feature in ["Age"]:
                # 使用 .loc 和 Min-Max 缩放处理异常值
                data.loc[:, feature] = data[feature].clip(min_val, max_val)
            else:
                # 删除包含异常值的行
                data = data[~data[feature].isin(outliers)]
    # print("\n异常值检测 (基于已知范围):")
    # for feature in numerical_features + categorical_features:
    #     if feature == "Age":
    #         min_val, max_val = 15, 18
    #     elif feature == "StudyTimeWeekly":
    #         min_val, max_val = 0, 20
    #     elif feature == "Absences":
    #         min_val, max_val = 0, 30
    #     elif feature == "GPA":
    #         min_val, max_val = 0, 4.0  # Corrected GPA range
    #     elif feature == "Gender":
    #         min_val, max_val = 0, 1
    #     elif feature == "Ethnicity":
    #         min_val, max_val = 0, 3
    #     elif feature == "ParentalEducation":
    #         min_val, max_val = 0, 4
    #     elif feature == "Tutoring":
    #         min_val, max_val = 0, 1
    #     elif feature == "ParentalInvolvement":
    #         min_val, max_val = 0, 4
    #     elif feature == "Extracurricular":
    #         min_val, max_val = 0, 1
    #     elif feature == "Sports":
    #         min_val, max_val = 0, 1
    #     elif feature == "Music":
    #         min_val, max_val = 0, 1
    #     elif feature == "Volunteering":
    #         min_val, max_val = 0, 1
    #     else:
    #         min_val, max_val = (
    #             data[feature].min(),
    #             data[feature].max(),
    #         )  # 如果没有明确范围，则使用数据的最小值和最大值
    #     outliers = data[(data[feature] < min_val) | (data[feature] > max_val)][feature]
    #     if not outliers.empty:
    #         print(f"{feature} 中的异常值: \n{outliers}")
    #     else:
    #         print(f"{feature} 中没有发现超出范围的异常值")

    return data


def show_data(data):
    # GPA 分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data["GPA"])
    plt.title("GPA分布", fontproperties=font)
    plt.xlabel("GPA", fontproperties=font)
    plt.ylabel("频率", fontproperties=font)
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font)
    plt.savefig("./assets/pic/gpa分布.png")
    plt.show()
    # 计算 GPA 的统计量
    gpa_mean = data["GPA"].mean()
    gpa_std = data["GPA"].std()
    gpa_ci = np.percentile(data["GPA"], [2.5, 97.5])
    print("\nGPA统计量：")
    print(f"  GPA均值: {gpa_mean:.3f}")
    print(f"  GPA标准差: {gpa_std:.3f}")
    print(f"  GPA的95%置信区间: [{gpa_ci[0]:.3f}, {gpa_ci[1]:.3f}]")

    plt.figure(figsize=(10, 6))

    sns.kdeplot(data["GPA"], color="blue", label="实际GPA", linewidth=2)
    plt.title("实际GPA分布")
    plt.xlabel("GPA")
    plt.ylabel("密度")
    plt.xlim(0, 4)
    # 显示图例
    plt.legend()

    # 显示图形
    plt.savefig("./assets/pic/gpa线.png")
    plt.show()


def description_wash():
    data = pd.read_csv("./assets/Student_performance_data.csv")
    data = data.drop("StudentID", axis=1)
    # data = data.drop("GradeClass", axis=1)

    # descriptive_stats(data)
    # describe_stats(data)
    data = clean_wash(data)
    # show_data(data)
    return data
