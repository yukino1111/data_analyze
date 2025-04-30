from config import *
from common import description_wash

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# --- 监督学习所需库 ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# --- Unsupervised Learning Imports ---
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
)

import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.cluster._kmeans"
)
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def preprocess_data(df, categorical_cols_to_encode):
    """对数值列标准化，对指定分类列进行独热编码"""
    df_processed = df.copy()
    numerical_features1 = ["GPA"]
    numerical_features2 = ["StudyTimeWeekly", "Absences", "Age"]
    if numerical_features1:  # 检查列表是否为空
        scaler = StandardScaler()
        df_processed[numerical_features1] = scaler.fit_transform(
            df[numerical_features1]
        )
        print(f"数值特征 {numerical_features1} 已进行标准化处理。")
    if numerical_features2:
        scaler = MinMaxScaler()
        df_processed[numerical_features2] = scaler.fit_transform(
            df[numerical_features2]
        )
        print(f"数值特征 {numerical_features2} 已进行归一化处理 (Min-Max 缩放)。")
    # 2. 独热编码分类特征
    if categorical_cols_to_encode:
        # drop_first=True 可以减少一个维度，避免多重共线性，常用于线性模型
        df_processed = pd.get_dummies(
            df_processed,
            columns=categorical_cols_to_encode,
            drop_first=False,
            dtype=int,
        )
        print(f"分类特征 {categorical_cols_to_encode} 已进行独热编码处理。")
    return df_processed


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """训练逻辑回归模型并评估"""
    print("\n--- 训练逻辑回归模型 ---")
    start_time = time.time()
    model = LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=grade_labels)
    cm = confusion_matrix(y_test, y_pred)
    training_time = end_time - start_time

    print(f"训练时间: {training_time:.4f} 秒")
    print(f"准确率: {accuracy:.4f}")

    return {
        "name": "Logistic Regression",
        "accuracy": accuracy,
        "report": report,
        "cm": cm,
        "time": training_time,
    }


def train_decision_tree(X_train, y_train, X_test, y_test):
    """训练决策树模型并评估"""
    print("\n--- 训练决策树模型 ---")
    start_time = time.time()
    # 设置 random_state 以便结果可复现，可以调整 max_depth 等防止过拟合
    model = DecisionTreeClassifier(random_state=42, max_depth=5)  # 示例：限制最大深度
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=grade_labels)
    cm = confusion_matrix(y_test, y_pred)
    training_time = end_time - start_time

    print(f"训练时间: {training_time:.4f} 秒")
    print(f"准确率: {accuracy:.4f}")

    return {
        "name": "Decision Tree",
        "accuracy": accuracy,
        "report": report,
        "cm": cm,
        "time": training_time,
    }


def train_svm(X_train, y_train, X_test, y_test, kernel="rbf"):
    """训练支持向量机(SVM)模型并评估，支持不同的核函数"""
    print(f"\n--- 训练支持向量机 (SVM) 模型 (Kernel: {kernel}) ---")
    start_time = time.time()
    model = SVC(kernel=kernel, random_state=42, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=grade_labels)
    cm = confusion_matrix(y_test, y_pred)
    training_time = end_time - start_time
    print(f"训练时间: {training_time:.4f} 秒")
    print(f"准确率: {accuracy:.4f}")
    return {
        "name": f"SVM ({kernel.capitalize()} Kernel)",
        "accuracy": accuracy,
        "report": report,
        "cm": cm,
        "time": training_time,
        "kernel": kernel,  # 保存核函数信息
    }


def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()  # 调整布局防止标签重叠
    plt.savefig(f"./assets/pic/{title}.png")
    plt.show()


def plot_comparison(results):
    """绘制不同算法性能对比图 (准确率和训练时间)"""
    names = [r["name"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    times = [r["time"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制准确率条形图
    color = "tab:blue"
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Accuracy", color=color)
    bars = ax1.bar(names, accuracies, color=color, alpha=0.6, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 1.05)  # 设置准确率范围

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom",
            ha="center",
        )  # va='bottom' 放在条形图上方

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Training Time (s)", color=color)
    ax2.plot(
        names, times, color=color, marker="o", linestyle="--", label="Training Time"
    )
    ax2.tick_params(axis="y", labelcolor=color)

    fig.suptitle("Algorithm Comparison: Accuracy and Training Time")  # 主标题
    fig.legend(
        loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
    )  # 合并图例
    plt.xticks(rotation=15)  # 轻微旋转X轴标签防止重叠
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，留出主标题空间
    plt.savefig(f"./assets/pic/Accuracy and Training Time.png")
    plt.show()


# 定义 GradeClass 标签映射 (用于报告和绘图) - 确保这个定义在调用它的函数之前
grade_labels_map = {
    0: "A (>=3.5)",
    1: "B (3.0-3.5)",
    2: "C (2.5-3.0)",
    3: "D (2.0-2.5)",
    4: "F (<2.0)",
}
grade_labels = []


# --- 3. 修改主函数流程 ---
def Supervised_Learning(data_processed1):
    print("\n==========================")
    print("   开始执行监督学习任务   ")
    print("==========================")
    data_processed = data_processed1.copy()
    data_processed = data_processed.drop("GPA", axis=1)
    # 1. 分离特征 (X) 和目标变量 (y)
    if "GradeClass" not in data_processed.columns:
        print("错误：目标变量 'GradeClass' 不在处理后的数据中！")
        return

    X = data_processed.drop("GradeClass", axis=1)
    y = data_processed["GradeClass"]

    global grade_labels
    unique_grades = sorted(y.unique())
    grade_labels = [grade_labels_map.get(g, f"Unknown({g})") for g in unique_grades]
    print(f"识别到的等级标签: {grade_labels}")

    # 2. 划分训练集和测试集
    # stratify=y 确保训练集和测试集中各类别比例与原始数据一致
    # random_state 保证每次划分结果相同，便于复现
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        print(
            f"\n数据已划分为训练集 ({X_train.shape[0]} 条) 和测试集 ({X_test.shape[0]} 条)"
        )
        print(f"特征数量: {X_train.shape[1]}")
    except ValueError as e:
        print(f"数据划分错误: {e}")
        print("请检查目标变量 'GradeClass' 是否存在或数据是否太少。")
        return

    # 3. 训练和评估各个模型
    results = []

    # 调用逻辑回归
    lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    results.append(lr_results)

    # 调用决策树
    dt_results = train_decision_tree(X_train, y_train, X_test, y_test)
    results.append(dt_results)

    # 训练不同核函数的 SVM 模型
    svm_kernels = ["linear", "poly", "rbf"]
    for kernel in svm_kernels:
        svm_results = train_svm(X_train, y_train, X_test, y_test, kernel=kernel)
        results.append(svm_results)

    # 4. 结果可视化
    print("\n--- 生成评估图表 ---")

    # 绘制每个模型的混淆矩阵
    for result in results:
        plot_confusion_matrix(
            result["cm"],
            classes=grade_labels,
            title=f"{result['name']} - Confusion Matrix",
        )
        # 打印详细分类报告
        print(f"\n--- {result['name']} 分类报告 ---")
        print(result["report"])

    # 绘制性能对比图
    plot_comparison(results)

    print("\n==========================")
    print("   监督学习任务执行完毕   ")
    print("==========================")

    return results


# def find_optimal_k_elbow(X, max_k=11, random_state=42, supposed_k=-1):
#     print("\n--- 寻找 K-Means 的最佳 K ---")
#     inertias = []
#     silhouette_scores = []
#     calinski_harabasz_scores = []
#     k_range = range(2, max_k)  # K 至少为 2 才有意义
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
#         kmeans.fit(X)
#         inertias.append(kmeans.inertia_)
#         # 计算轮廓系数，需要所有数据点的标签
#         labels = kmeans.labels_
#         silhouette_scores.append(silhouette_score(X, labels))
#         calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
#     # 绘制肘部图
#     plt.figure(figsize=(13, 6))  # 增加图像宽度，方便显示多个子图
#     plt.subplot(1, 2, 1)  # 1 行 3 列，选择第 1 个子图
#     plt.plot(k_range, inertias, marker="o")
#     plt.title("Elbow Method for Optimal K")
#     plt.xlabel("Number of Clusters (K)")
#     plt.ylabel("Inertia (WCSS)")
#     plt.xticks(k_range)
#     plt.grid(True)
#     # 绘制轮廓系数图
#     plt.subplot(1, 2, 2)  # 1 行 3 列，选择第 2 个子图
#     plt.plot(k_range, silhouette_scores, marker="o")
#     plt.title("Silhouette Score for Optimal K")
#     plt.xlabel("Number of Clusters (K)")
#     plt.ylabel("Silhouette Score")
#     plt.xticks(k_range)
#     plt.grid(True)
#     # 绘制 Calinski-Harabasz 指数图
#     # plt.subplot(1, 3, 3)  # 1 行 3 列，选择第 3 个子图
#     # plt.plot(k_range, calinski_harabasz_scores, marker="o")
#     # plt.title("Calinski-Harabasz Index for Optimal K")
#     # plt.xlabel("Number of Clusters (K)")
#     # plt.ylabel("Calinski-Harabasz Index")
#     # plt.xticks(k_range)
#     # plt.grid(True)
#     # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
#     plt.savefig(f"./assets/pic/find_k_{supposed_k}.png")
#     plt.show()
#     print("请观察上面的肘部图、轮廓系数图和指数图，综合确定一个合适的 K 值。")
#     return k_range, inertias, silhouette_scores, calinski_harabasz_scores


# def run_kmeans(X, n_clusters, random_state=42):
#     print(f"\n--- 运行 K-Means (K={n_clusters}) ---")
#     start_time = time.time()
#     kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
#     labels = kmeans.fit_predict(X)
#     end_time = time.time()
#     training_time = end_time - start_time

#     try:
#         score = silhouette_score(X, labels)
#         print(f"轮廓系数 (Silhouette Score): {score:.4f}")
#     except ValueError:
#         score = -1  # Or np.nan
#         print("轮廓系数计算失败 (可能只有一个聚类被找到)")

#     print(f"训练时间: {training_time:.4f} 秒")
#     return {
#         "name": "K-Means",
#         "labels": labels,
#         "score": score,
#         "time": training_time,
#         "n_clusters": n_clusters,
#     }


# def run_hierarchical(X, n_clusters):
#     """运行层次聚类 (Agglomerative)"""
#     print(f"\n--- 运行层次聚类 (n_clusters={n_clusters}) ---")
#     start_time = time.time()
#     # 使用 'ward' 链接，它倾向于寻找方差最小的簇
#     agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
#     labels = agg_clustering.fit_predict(X)
#     end_time = time.time()
#     training_time = end_time - start_time

#     try:
#         score = silhouette_score(X, labels)
#         print(f"轮廓系数 (Silhouette Score): {score:.4f}")
#     except ValueError:
#         score = -1
#         print("轮廓系数计算失败 (可能只有一个聚类被找到)")

#     print(f"训练时间: {training_time:.4f} 秒")
#     return {
#         "name": "Hierarchical (Ward)",
#         "labels": labels,
#         "score": score,
#         "time": training_time,
#         "n_clusters": n_clusters,
#     }


# def run_dbscan(X, eps=0.5, min_samples=5):
#     """运行 DBSCAN 聚类"""
#     print(f"\n--- 运行 DBSCAN (eps={eps}, min_samples={min_samples}) ---")
#     start_time = time.time()
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     labels = dbscan.fit_predict(X)
#     end_time = time.time()
#     training_time = end_time - start_time

#     # 计算轮廓系数时要排除噪声点 (label=-1)
#     # 确保至少有 2 个簇（不包括噪声）才能计算轮廓系数
#     unique_labels = set(labels)
#     n_clusters_found = len(unique_labels) - (
#         1 if -1 in labels else 0
#     )  # 减去噪声标签（如果存在）
#     print(f"找到的聚类数量 (不含噪声): {n_clusters_found}")
#     print(f"噪声点数量: {np.sum(labels == -1)}")

#     score = -1  # 默认值
#     if n_clusters_found >= 2:
#         try:
#             # 只在非噪声点上计算轮廓系数
#             mask = labels != -1
#             if np.sum(mask) > 0:  # 确保有非噪声点
#                 score = silhouette_score(X[mask], labels[mask])
#                 print(f"轮廓系数 (Silhouette Score, 仅非噪声点): {score:.4f}")
#             else:
#                 print("没有非噪声点，无法计算轮廓系数。")
#         except ValueError as e:
#             print(f"轮廓系数计算失败: {e}")
#             # 可能是因为所有点都被分到了一个簇中（即使排除了噪声）
#     elif n_clusters_found == 1:
#         print("只找到一个聚类 (不含噪声)，无法计算轮廓系数。")
#     else:  # n_clusters_found == 0
#         print("未找到有效聚类 (全是噪声?)，无法计算轮廓系数。")

#     print(f"处理时间: {training_time:.4f} 秒")
#     return {
#         "name": f"DBSCAN (eps={eps}, min={min_samples})",
#         "labels": labels,
#         "score": score,
#         "time": training_time,
#         "n_clusters": n_clusters_found,
#     }


# def apply_pca(X, n_components=2, random_state=42):
#     """应用 PCA 降维"""
#     print(f"\n--- 应用 PCA 降维至 {n_components} 维 ---")
#     pca = PCA(n_components=n_components, random_state=random_state)
#     X_reduced = pca.fit_transform(X)
#     print(f"解释的总方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
#     return X_reduced, pca


# def apply_tsne(X, n_components=2, random_state=42, perplexity=30.0):
#     """应用 t-SNE 降维"""
#     print(f"\n--- 应用 t-SNE 降维至 {n_components} 维 (这可能需要一些时间) ---")
#     start_time = time.time()
#     # 检查样本数量是否足够支持 perplexity
#     n_samples = X.shape[0]
#     effective_perplexity = min(
#         perplexity, n_samples - 1
#     )  # Perplexity 不能大于 n_samples - 1
#     if effective_perplexity != perplexity:
#         print(
#             f"警告: Perplexity ({perplexity}) 大于 n_samples-1 ({n_samples-1}). 使用 {effective_perplexity} 代替。"
#         )
#     if effective_perplexity <= 0:
#         print("错误: 样本数量不足以运行 t-SNE。")
#         return None, None  # 返回 None 表示失败

#     tsne = TSNE(
#         n_components=n_components,
#         random_state=random_state,
#         perplexity=effective_perplexity,
#         max_iter=300,
#     )  # n_iter 可以调整
#     X_reduced = tsne.fit_transform(X)
#     end_time = time.time()
#     print(f"t-SNE 降维耗时: {end_time - start_time:.2f} 秒")
#     return X_reduced, tsne


# def plot_clusters_2d(X_reduced, labels, algorithm_name, dr_name, n_clusters_found):
#     """在 2D 空间中绘制聚类结果"""
#     plt.figure(figsize=(10, 8))
#     unique_labels = sorted(list(set(labels)))
#     colors = plt.cm.viridis(
#         np.linspace(0, 1, len(unique_labels))
#     )  # 使用 viridis 色彩映射

#     for k, col in zip(unique_labels, colors):
#         if k == -1:
#             # 噪声点用灰色、小标记表示
#             col = [0.5, 0.5, 0.5, 1]  # 灰色
#             marker = "x"
#             markersize = 6
#             label = "Noise"
#         else:
#             marker = "o"
#             markersize = 8
#             label = f"Cluster {k}"

#         class_member_mask = labels == k
#         xy = X_reduced[class_member_mask]
#         plt.scatter(
#             xy[:, 0],
#             xy[:, 1],
#             s=markersize,
#             c=[col],
#             marker=marker,
#             label=label,
#             alpha=0.7,
#         )  # 使用列表包装颜色以避免警告

#     plt.title(
#         f"{algorithm_name} Clusters (Projected by {dr_name})\nFound {n_clusters_found} clusters"
#     )
#     plt.xlabel(f"{dr_name} Component 1")
#     plt.ylabel(f"{dr_name} Component 2")
#     plt.legend(loc="best", fontsize="small")
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.savefig(
#         f"./assets/pic/{algorithm_name} Clusters (Projected by {dr_name})_{n_clusters_found}.png"
#     )
#     plt.show()


# def plot_unsupervised_comparison(results, supposed_k=-1):
#     """绘制不同聚类算法性能对比图 (轮廓系数和时间)"""
#     # 过滤掉 score 为 -1 (计算失败) 的结果，避免绘图错误
#     valid_results = [r for r in results if r["score"] != -1 and r["score"] is not None]
#     if not valid_results:
#         print("没有有效的轮廓系数可供比较。")
#         return

#     names = [r["name"] for r in valid_results]
#     scores = [r["score"] for r in valid_results]
#     times = [r["time"] for r in valid_results]

#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     # 绘制轮廓系数条形图
#     color = "tab:green"
#     ax1.set_xlabel("Algorithm")
#     ax1.set_ylabel("Silhouette Score", color=color)
#     bars = ax1.bar(names, scores, color=color, alpha=0.6, label="Silhouette Score")
#     ax1.tick_params(axis="y", labelcolor=color)
#     # 调整 Y 轴范围，轮廓系数在 [-1, 1] 之间
#     ax1.set_ylim(
#         min(min(scores) - 0.1, -0.2), max(max(scores) + 0.1, 0.5)
#     )  # 动态调整，但至少包含部分负数区域

#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width() / 2.0,
#             yval,
#             f"{yval:.3f}",
#             va="bottom" if yval >= 0 else "top",
#             ha="center",
#         )

#     ax2 = ax1.twinx()
#     color = "tab:red"
#     ax2.set_ylabel("Processing Time (s)", color=color)
#     ax2.plot(
#         names, times, color=color, marker="o", linestyle="--", label="Processing Time"
#     )
#     ax2.tick_params(axis="y", labelcolor=color)
#     ax2.set_ylim(0, max(times) * 1.1)

#     fig.suptitle("Clustering Algorithm Comparison: Silhouette Score and Time")
#     fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
#     plt.xticks(rotation=15)
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.savefig(f"./assets/pic/Clustering Algorithm Comparison_{supposed_k}.png")
#     plt.show()


# def Unsupervised_Learning(data_processed2, k):
#     """执行无监督学习流程：聚类、降维和评估"""
#     print("\n=============================")
#     print("   开始执行无监督学习任务   ")
#     print("=============================")
#     data_processed = data_processed2.copy()
#     # 1. 准备数据：通常不使用目标变量
#     if "GradeClass" in data_processed.columns:
#         X_unsupervised = data_processed.drop("GradeClass", axis=1)
#         print("已移除 'GradeClass' 用于无监督学习。")
#     elif "GPA" in data_processed.columns:
#         X_unsupervised = data_processed.drop("GPA", axis=1)
#         print("已移除 'GPA' 用于无监督学习。")
#     else:
#         X_unsupervised = data_processed.copy()
#         print("未找到 'GradeClass'，使用所有提供的特征进行无监督学习。")

#     if X_unsupervised.empty:
#         print("错误：用于无监督学习的数据为空！")
#         return

#     print(f"用于聚类的数据维度: {X_unsupervised.shape}")

#     # 2. K-Means: 先用肘部法则确定 K
#     find_optimal_k_elbow(X_unsupervised, max_k=26, supposed_k=k)
#     # *** 手动选择 K ***
#     # 基于上面的肘部图，或者你的领域知识，选择一个 K 值
#     # 例如，如果图表在 K=4 或 K=5 处有拐点，或者你知道有 5 个等级
#     chosen_k = k
#     print(f"\n*** 基于肘部图或先验知识，选择 K = {chosen_k} ***")

#     results = []

#     # # 运行 K-Means
#     kmeans_results = run_kmeans(X_unsupervised, n_clusters=chosen_k)
#     results.append(kmeans_results)

#     # # 3. 运行层次聚类 (使用与 K-Means 相同的 K 以便比较)
#     hierarchical_results = run_hierarchical(X_unsupervised, n_clusters=chosen_k)
#     results.append(hierarchical_results)

#     # 4. 运行 DBSCAN
#     # DBSCAN 的参数 eps 和 min_samples 对结果影响很大，且依赖数据
#     # 这里的 0.5 和 5 是示例值，对于标准化后的数据可能是一个起点
#     # 你可能需要根据数据特性和轮廓系数反馈来调整这些值
#     # 例如，如果噪声点太多，尝试增大 eps 或 min_samples
#     # 如果簇太少或太大，尝试减小 eps
#     dbscan_eps = 1.25  # <--- 示例值，需要调整
#     dbscan_min_samples = X_unsupervised.shape[1] + 1  # <--- 示例值，需要调整
#     dbscan_results = run_dbscan(
#         X_unsupervised, eps=dbscan_eps, min_samples=dbscan_min_samples
#     )
#     results.append(dbscan_results)

#     # 5. 降维用于可视化
#     # 应用 PCA
#     X_pca, pca_model = apply_pca(X_unsupervised)
#     # 应用 t-SNE (如果数据量大，这步会比较慢)
#     X_tsne, tsne_model = apply_tsne(X_unsupervised)  # 使用默认 perplexity

#     # 6. 可视化聚类结果 (使用 PCA 和 t-SNE 降维后的数据)
#     print("\n--- 生成聚类结果可视化图表 ---")
#     if X_pca is not None:
#         for result in results:
#             plot_clusters_2d(
#                 X_pca, result["labels"], result["name"], "PCA", result["n_clusters"]
#             )

#     if X_tsne is not None:
#         for result in results:
#             # DBSCAN 可能找到的簇数与 K-Means/Hierarchical 不同
#             plot_clusters_2d(
#                 X_tsne, result["labels"], result["name"], "t-SNE", result["n_clusters"]
#             )
#     else:
#         print("t-SNE 降维失败，跳过 t-SNE 可视化。")

#     # 7. 绘制性能对比图
#     print("\n--- 生成算法性能对比图 ---")
#     plot_unsupervised_comparison(results, supposed_k=k)

#     print("\n=============================")
#     print("   无监督学习任务执行完毕   ")
#     print("=============================")


#     return results  # 返回结果供后续分析
def find_optimal_k_elbow(
    X, max_k=11, random_state=42, supposed_k=-1, dr_name="Original"
):
    print(f"\n--- 在 {dr_name} 数据上寻找 K-Means 的最佳 K ---")
    inertias = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    k_range = range(2, max_k)  # K 至少为 2 才有意义
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        # 计算轮廓系数，需要所有数据点的标签
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
    # 绘制肘部图
    plt.figure(figsize=(13, 6))  # 增加图像宽度，方便显示多个子图
    plt.subplot(1, 2, 1)  # 1 行 3 列，选择第 1 个子图
    plt.plot(k_range, inertias, marker="o")
    plt.title(f"Elbow Method for Optimal K ({dr_name})")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.xticks(k_range)
    plt.grid(True)
    # 绘制轮廓系数图
    plt.subplot(1, 2, 2)  # 1 行 3 列，选择第 2 个子图
    plt.plot(k_range, silhouette_scores, marker="o")
    plt.title(f"Silhouette Score for Optimal K ({dr_name})")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_range)
    plt.grid(True)
    # 绘制 Calinski-Harabasz 指数图
    # plt.subplot(1, 3, 3)  # 1 行 3 列，选择第 3 个子图
    # plt.plot(k_range, calinski_harabasz_scores, marker="o")
    # plt.title(f"Calinski-Harabasz Index for Optimal K ({dr_name})")
    # plt.xlabel("Number of Clusters (K)")
    # plt.ylabel("Calinski-Harabasz Index")
    # plt.xticks(k_range)
    # plt.grid(True)
    # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig(f"./assets/pic/find_k_{dr_name}_{supposed_k}.png")
    plt.show()
    print(
        f"请观察上面的 {dr_name} 数据上的肘部图、轮廓系数图和指数图，综合确定一个合适的 K 值。"
    )
    return k_range, inertias, silhouette_scores, calinski_harabasz_scores


def run_kmeans(X, n_clusters, random_state=42, dr_name="Original"):
    print(f"\n--- 在 {dr_name} 数据上运行 K-Means (K={n_clusters}) ---")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    end_time = time.time()
    training_time = end_time - start_time

    try:
        score = silhouette_score(X, labels)
        print(f"轮廓系数 (Silhouette Score): {score:.4f}")
    except ValueError:
        score = -1  # Or np.nan
        print("轮廓系数计算失败 (可能只有一个聚类被找到)")

    print(f"训练时间: {training_time:.4f} 秒")
    return {
        "name": f"K-Means ({dr_name})",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters,
        "dr_name": dr_name,  # 添加降维方法名称
    }


def run_hierarchical(X, n_clusters, dr_name="Original"):
    """运行层次聚类 (Agglomerative)"""
    print(f"\n--- 在 {dr_name} 数据上运行层次聚类 (n_clusters={n_clusters}) ---")
    start_time = time.time()
    # 使用 'ward' 链接，它倾向于寻找方差最小的簇
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = agg_clustering.fit_predict(X)
    end_time = time.time()
    training_time = end_time - start_time

    try:
        score = silhouette_score(X, labels)
        print(f"轮廓系数 (Silhouette Score): {score:.4f}")
    except ValueError:
        score = -1
        print("轮廓系数计算失败 (可能只有一个聚类被找到)")

    print(f"训练时间: {training_time:.4f} 秒")
    return {
        "name": f"Hierarchical (Ward, {dr_name})",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters,
        "dr_name": dr_name,  # 添加降维方法名称
    }


def run_dbscan(X, eps=0.5, min_samples=5, dr_name="Original"):
    """运行 DBSCAN 聚类"""
    print(
        f"\n--- 在 {dr_name} 数据上运行 DBSCAN (eps={eps}, min_samples={min_samples}) ---"
    )
    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    end_time = time.time()
    training_time = end_time - start_time

    # 计算轮廓系数时要排除噪声点 (label=-1)
    # 确保至少有 2 个簇（不包括噪声）才能计算轮廓系数
    unique_labels = set(labels)
    n_clusters_found = len(unique_labels) - (
        1 if -1 in labels else 0
    )  # 减去噪声标签（如果存在）
    print(f"找到的聚类数量 (不含噪声): {n_clusters_found}")
    print(f"噪声点数量: {np.sum(labels == -1)}")

    score = -1  # 默认值
    if n_clusters_found >= 2:
        try:
            # 只在非噪声点上计算轮廓系数
            mask = labels != -1
            if np.sum(mask) > 0:  # 确保有非噪声点
                score = silhouette_score(X[mask], labels[mask])
                print(f"轮廓系数 (Silhouette Score, 仅非噪声点): {score:.4f}")
            else:
                print("没有非噪声点，无法计算轮廓系数。")
        except ValueError as e:
            print(f"轮廓系数计算失败: {e}")
            # 可能是因为所有点都被分到了一个簇中（即使排除了噪声）
    elif n_clusters_found == 1:
        print("只找到一个聚类 (不含噪声)，无法计算轮廓系数。")
    else:  # n_clusters_found == 0
        print("未找到有效聚类 (全是噪声?)，无法计算轮廓系数。")

    print(f"处理时间: {training_time:.4f} 秒")
    return {
        "name": f"DBSCAN (eps={eps}, min={min_samples}, {dr_name})",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters_found,
        "dr_name": dr_name,  # 添加降维方法名称
    }


def apply_pca(X, n_components=2, random_state=42):
    """应用 PCA 降维"""
    print(f"\n--- 应用 PCA 降维至 {n_components} 维 ---")
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)
    print(f"解释的总方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
    return X_reduced, pca


def apply_tsne(X, n_components=2, random_state=42, perplexity=30.0):
    """应用 t-SNE 降维"""
    print(f"\n--- 应用 t-SNE 降维至 {n_components} 维 (这可能需要一些时间) ---")
    start_time = time.time()
    # 检查样本数量是否足够支持 perplexity
    n_samples = X.shape[0]
    effective_perplexity = min(
        perplexity, n_samples - 1
    )  # Perplexity 不能大于 n_samples - 1
    if effective_perplexity != perplexity:
        print(
            f"警告: Perplexity ({perplexity}) 大于 n_samples-1 ({n_samples-1}). 使用 {effective_perplexity} 代替。"
        )
    if effective_perplexity <= 0:
        print("错误: 样本数量不足以运行 t-SNE。")
        return None, None  # 返回 None 表示失败

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=effective_perplexity,
        max_iter=300,
    )  # n_iter 可以调整
    X_reduced = tsne.fit_transform(X)
    end_time = time.time()
    print(f"t-SNE 降维耗时: {end_time - start_time:.2f} 秒")
    return X_reduced, tsne


def plot_clusters_2d(X_reduced, labels, algorithm_name, dr_name, n_clusters_found):
    """在 2D 空间中绘制聚类结果"""
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(list(set(labels)))
    colors = plt.cm.viridis(
        np.linspace(0, 1, len(unique_labels))
    )  # 使用 viridis 色彩映射

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪声点用灰色、小标记表示
            col = [0.5, 0.5, 0.5, 1]  # 灰色
            marker = "x"
            markersize = 6
            label = "Noise"
        else:
            marker = "o"
            markersize = 8
            label = f"Cluster {k}"

        class_member_mask = labels == k
        xy = X_reduced[class_member_mask]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=markersize,
            c=[col],
            marker=marker,
            label=label,
            alpha=0.7,
        )  # 使用列表包装颜色以避免警告

    plt.title(
        f"{algorithm_name} Clusters (Projected by {dr_name})\nFound {n_clusters_found} clusters"
    )
    plt.xlabel(f"{dr_name} Component 1")
    plt.ylabel(f"{dr_name} Component 2")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(
        f"./assets/pic/{algorithm_name} Clusters (Projected by {dr_name})_{n_clusters_found}.png"
    )
    plt.show()


# def plot_unsupervised_comparison(results, supposed_k=-1):
#     """绘制不同聚类算法性能对比图 (轮廓系数和时间)"""
#     # 过滤掉 score 为 -1 (计算失败) 的结果，避免绘图错误
#     valid_results = [r for r in results if r["score"] != -1 and r["score"] is not None]
#     if not valid_results:
#         print("没有有效的轮廓系数可供比较。")
#         return

#     # 按降维方法分组结果
#     results_by_dr = {}
#     for r in valid_results:
#         dr_name = r.get("dr_name", "Original")  # 获取降维方法名称
#         if dr_name not in results_by_dr:
#             results_by_dr[dr_name] = []
#         results_by_dr[dr_name].append(r)

#     for dr_name, dr_results in results_by_dr.items():
#         names = [r["name"] for r in dr_results]
#         scores = [r["score"] for r in dr_results]
#         times = [r["time"] for r in dr_results]

#         fig, ax1 = plt.subplots(figsize=(10, 6))

#         # 绘制轮廓系数条形图
#         color = "tab:green"
#         ax1.set_xlabel("Algorithm")
#         ax1.set_ylabel("Silhouette Score", color=color)
#         bars = ax1.bar(names, scores, color=color, alpha=0.6, label="Silhouette Score")
#         ax1.tick_params(axis="y", labelcolor=color)
#         # 调整 Y 轴范围，轮廓系数在 [-1, 1] 之间
#         ax1.set_ylim(
#             min(min(scores) - 0.1, -0.2), max(max(scores) + 0.1, 0.5)
#         )  # 动态调整，但至少包含部分负数区域

#         for bar in bars:
#             yval = bar.get_height()
#             plt.text(
#                 bar.get_x() + bar.get_width() / 2.0,
#                 yval,
#                 f"{yval:.3f}",
#                 va="bottom" if yval >= 0 else "top",
#                 ha="center",
#             )

#         ax2 = ax1.twinx()
#         color = "tab:red"
#         ax2.set_ylabel("Processing Time (s)", color=color)
#         ax2.plot(
#             names,
#             times,
#             color=color,
#             marker="o",
#             linestyle="--",
#             label="Processing Time",
#         )
#         ax2.tick_params(axis="y", labelcolor=color)
#         ax2.set_ylim(0, max(times) * 1.1)

#         fig.suptitle(
#             f"Clustering Algorithm Comparison on {dr_name} Data: Silhouette Score and Time"
#         )
#         fig.legend(
#             loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes
#         )
#         plt.xticks(rotation=15)
#         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#         plt.savefig(
#             f"./assets/pic/Clustering Algorithm Comparison_{dr_name}_{supposed_k}.png"
#         )
#         plt.show()


# def Unsupervised_Learning(data_processed2, pca_k,tsne_k,k):
#     """执行无监督学习流程：降维、聚类和评估"""
#     print("\n=============================")
#     print("   开始执行无监督学习任务   ")
#     print("=============================")
#     data_processed = data_processed2.copy()
#     # 1. 准备数据：通常不使用目标变量
#     if "GradeClass" in data_processed.columns:
#         X_unsupervised = data_processed.drop("GradeClass", axis=1)
#         print("已移除 'GradeClass' 用于无监督学习。")
#     elif "GPA" in data_processed.columns:
#         X_unsupervised = data_processed.drop("GPA", axis=1)
#         print("已移除 'GPA' 用于无监督学习。")
#     else:
#         X_unsupervised = data_processed.copy()
#         print("未找到 'GradeClass'，使用所有提供的特征进行无监督学习。")

#     if X_unsupervised.empty:
#         print("错误：用于无监督学习的数据为空！")
#         return

#     print(f"用于无监督学习的数据维度: {X_unsupervised.shape}")

#     # 2. 降维
#     # 应用 PCA
#     X_pca, pca_model = apply_pca(X_unsupervised)
#     # 应用 t-SNE (如果数据量大，这步会比较慢)
#     X_tsne, tsne_model = apply_tsne(X_unsupervised)  # 使用默认 perplexity

#     # 3. 在降维后的数据上寻找 K-Means 的最佳 K (可选，可以在每个降维结果上都运行)
#     # 这里为了简化，只在 PCA 结果上运行肘部法则作为示例
#     chosen_k = k  # 默认使用传入的 k
#     if X_pca is not None:
#         find_optimal_k_elbow(X_pca, max_k=26, supposed_k=pca_k, dr_name="PCA")
#         # *** 手动选择 K ***
#         # 基于上面的肘部图，或者你的领域知识，选择一个 K 值
#         # chosen_k = ... # 如果需要根据 PCA 结果重新选择 K，在这里修改
#         print(f"\n*** 基于 PCA 数据上的肘部图或先验知识，选择 K = {pca_k} ***")
#     # if X_tsne is not None:
#     #     find_optimal_k_elbow(X_tsne, max_k=26, supposed_k=tsne_k, dr_name="t-SNE")
#     #     # *** 手动选择 K ***
#     #     # chosen_k = ... # 如果需要根据 t-SNE 结果重新选择 K，在这里修改
#     #     print(f"\n*** 基于 t-SNE 数据上的肘部图或先验知识，选择 K = {tsne_k} ***")
#     else:
#         print(f"\n*** 未进行降维，使用默认 K = {k} ***")

#     results = []

#     # 4. 在降维后的数据上运行聚类算法
#     if X_pca is not None:
#         print("\n--- 在 PCA 降维数据上运行聚类算法 ---")
#         # 运行 K-Means
#         kmeans_pca_results = run_kmeans(X_pca, n_clusters=pca_k, dr_name="PCA")
#         results.append(kmeans_pca_results)

#         # 运行层次聚类
#         hierarchical_pca_results = run_hierarchical(
#             X_pca, n_clusters=pca_k, dr_name="PCA"
#         )
#         results.append(hierarchical_pca_results)

#         # 运行 DBSCAN (参数可能需要针对 PCA 数据调整)
#         dbscan_pca_eps = 0.5  # <--- 示例值，需要调整
#         dbscan_pca_min_samples = 5  # <--- 示例值，需要调整
#         dbscan_pca_results = run_dbscan(
#             X_pca, eps=dbscan_pca_eps, min_samples=dbscan_pca_min_samples, dr_name="PCA"
#         )
#         results.append(dbscan_pca_results)

#     if X_tsne is not None:
#         print("\n--- 在 t-SNE 降维数据上运行聚类算法 ---")
#         # 运行 K-Means
#         kmeans_tsne_results = run_kmeans(X_tsne, n_clusters=tsne_k, dr_name="t-SNE")
#         results.append(kmeans_tsne_results)

#         # 运行层次聚类
#         hierarchical_tsne_results = run_hierarchical(
#             X_tsne, n_clusters=tsne_k, dr_name="t-SNE"
#         )
#         results.append(hierarchical_tsne_results)

#         # 运行 DBSCAN (参数可能需要针对 t-SNE 数据调整)
#         # t-SNE 后的数据通常没有明确的密度概念，DBSCAN 可能效果不佳，参数更难选择
#         dbscan_tsne_eps = 0.5  # <--- 示例值，需要调整
#         dbscan_tsne_min_samples = 5  # <--- 示例值，需要调整
#         dbscan_tsne_results = run_dbscan(
#             X_tsne,
#             eps=dbscan_tsne_eps,
#             min_samples=dbscan_tsne_min_samples,
#             dr_name="t-SNE",
#         )
#         results.append(dbscan_tsne_results)

#     # 5. 可视化聚类结果 (使用降维后的数据和对应的聚类标签)
#     print("\n--- 生成聚类结果可视化图表 ---")
#     if X_pca is not None:
#         for result in results:
#             if result["dr_name"] == "PCA":  # 只绘制在 PCA 数据上聚类的结果
#                 plot_clusters_2d(
#                     X_pca, result["labels"], result["name"], "PCA", result["n_clusters"]
#                 )

#     if X_tsne is not None:
#         for result in results:
#             if result["dr_name"] == "t-SNE":  # 只绘制在 t-SNE 数据上聚类的结果
#                 plot_clusters_2d(
#                     X_tsne,
#                     result["labels"],
#                     result["name"],
#                     "t-SNE",
#                     result["n_clusters"],
#                 )
#     else:
#         print("t-SNE 降维失败，跳过 t-SNE 可视化。")

#     # 6. 绘制性能对比图
#     print("\n--- 生成算法性能对比图 ---")
#     plot_unsupervised_comparison(results, supposed_k=chosen_k)

#     print("\n=============================")
#     print("   无监督学习任务执行完毕   ")
#     print("=============================")

#     return results


SAVE_DIR = "./assets/pic/unsupervised"
# 确保保存目录存在
os.makedirs(SAVE_DIR, exist_ok=True)
# 随机状态，保证结果可复现
RANDOM_STATE = 42
# 寻找 K 值的最大范围
MAX_K_SEARCH = 15  # 可以根据需要调整


# --- 辅助函数 ---
def clean_filename(name):
    """清理文件名中的非法字符"""
    # 移除括号和逗号，替换空格和特殊字符
    name = name.replace("(", "").replace(")", "").replace(",", "")
    name = name.replace(" ", "_").replace("=", "").replace(":", "")
    # 可以添加更多需要替换的字符
    return name


def find_optimal_k_elbow(X, max_k, target_k, dr_name="Original"):
    """
    使用肘部法则和轮廓系数寻找 K-Means 的最佳 K 值，并保存图像。
    (样式已还原)
    """
    print(f"\n--- 在 {dr_name} 数据上寻找 K-Means 的最佳 K (目标 K: {target_k}) ---")
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k)
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            if len(set(labels)) > 1:
                 silhouette_scores.append(silhouette_score(X, labels))
            else:
                 silhouette_scores.append(-1)
        except Exception as e:
            print(f"计算 K={k} 时出错: {e}")
            inertias.append(np.nan)
            silhouette_scores.append(np.nan)
    plt.figure(figsize=(13, 6)) # 使用之前的尺寸
    # 肘部图
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker="o") # 使用之前的简单样式
    plt.title(f"Elbow Method for Optimal K ({dr_name})")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.xticks(k_range)
    plt.grid(True)
    # 标记目标 K (保留这个有用特性)
    # if target_k in k_range:
    #     target_idx = list(k_range).index(target_k)
    #     if not np.isnan(inertias[target_idx]):
    #         plt.scatter(target_k, inertias[target_idx], color='red', s=100, zorder=5, label=f'Target K={target_k}')
    #         plt.legend() # 只有标记时显示图例
    # 轮廓系数图
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker="o") # 使用之前的简单样式
    plt.title(f"Silhouette Score for Optimal K ({dr_name})")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_range)
    plt.grid(True)
    # 标记目标 K
    # if target_k in k_range:
    #     target_idx = list(k_range).index(target_k)
    #     if not np.isnan(silhouette_scores[target_idx]):
    #         plt.scatter(target_k, silhouette_scores[target_idx], color='red', s=100, zorder=5, label=f'Target K={target_k}')
    #         plt.legend() # 只有标记时显示图例
    plt.tight_layout() # 使用 tight_layout 自动调整
    filename = f"find_k_{dr_name}_targetK{target_k}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"寻找 K 值图像已保存至: {save_path}")
    plt.show()
    print(f"请观察 {dr_name} 数据的肘部图和轮廓系数图，评估目标 K={target_k} 的合理性。")

def run_kmeans(X, n_clusters, dr_name="Original"):
    """运行 K-Means 聚类并返回结果。"""
    print(f"\n--- 在 {dr_name} 数据上运行 K-Means (K={n_clusters}) ---")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    try:
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        print(f"轮廓系数 (Silhouette Score): {score:.4f}")
    except Exception as e:
        print(f"K-Means 运行或评分失败: {e}")
        labels = np.full(X.shape[0], -1)  # 标记为失败
        score = -1
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练时间: {training_time:.4f} 秒")
    return {
        "name": f"K-Means",  # 基础名称，dr_name 会在外部添加
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters,  # 目标簇数
        "dr_name": dr_name,
    }


def run_hierarchical(X, n_clusters, dr_name="Original"):
    """运行层次聚类 (Ward) 并返回结果。"""
    print(f"\n--- 在 {dr_name} 数据上运行 Hierarchical (Ward, K={n_clusters}) ---")
    start_time = time.time()
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    try:
        labels = agg_clustering.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        print(f"轮廓系数 (Silhouette Score): {score:.4f}")
    except Exception as e:
        print(f"Hierarchical 运行或评分失败: {e}")
        labels = np.full(X.shape[0], -1)
        score = -1
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练时间: {training_time:.4f} 秒")
    return {
        "name": f"Hierarchical_Ward",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters,  # 目标簇数
        "dr_name": dr_name,
    }


def run_dbscan(X, params, dr_name="Original"):
    """运行 DBSCAN 聚类并返回结果。"""
    eps = params.get("eps", 0.5)
    min_samples = params.get("min_samples", 5)
    print(
        f"\n--- 在 {dr_name} 数据上运行 DBSCAN (eps={eps}, min_samples={min_samples}) ---"
    )
    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    try:
        labels = dbscan.fit_predict(X)
        unique_labels = set(labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in labels else 0)
        print(f"找到的聚类数量 (不含噪声): {n_clusters_found}")
        print(f"噪声点数量: {np.sum(labels == -1)}")
        score = -1
        if n_clusters_found >= 2:
            mask = labels != -1
            if np.sum(mask) > 0:
                score = silhouette_score(X[mask], labels[mask])
                print(f"轮廓系数 (Silhouette Score, 仅非噪声点): {score:.4f}")
            else:
                print("没有非噪声点，无法计算轮廓系数。")
        elif n_clusters_found == 1:
            print("只找到一个聚类 (不含噪声)，无法计算轮廓系数。")
        else:
            print("未找到有效聚类，无法计算轮廓系数。")
    except Exception as e:
        print(f"DBSCAN 运行或评分失败: {e}")
        labels = np.full(X.shape[0], -1)
        score = -1
        n_clusters_found = 0
    end_time = time.time()
    training_time = end_time - start_time
    print(f"处理时间: {training_time:.4f} 秒")
    return {
        "name": f"DBSCAN_eps{eps}_min{min_samples}",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters_found,  # 实际找到的簇数
        "dr_name": dr_name,
    }


def apply_pca(X, n_components=2):
    """应用 PCA 降维。"""
    print(f"\n--- 应用 PCA 降维至 {n_components} 维 ---")
    try:
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        X_reduced = pca.fit_transform(X)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"解释的总方差比例: {explained_variance:.4f}")
        return X_reduced, pca, explained_variance
    except Exception as e:
        print(f"PCA 降维失败: {e}")
        return None, None, 0


def apply_tsne(X, n_components=2, perplexity=30.0):
    """应用 t-SNE 降维。"""
    print(f"\n--- 应用 t-SNE 降维至 {n_components} 维 (Perplexity: {perplexity}) ---")
    start_time = time.time()
    n_samples = X.shape[0]
    # 确保 perplexity 合理
    effective_perplexity = min(perplexity, max(1, n_samples - 1))
    if effective_perplexity != perplexity:
        print(
            f"警告: Perplexity ({perplexity}) 调整为 {effective_perplexity} (需小于样本数 {n_samples})."
        )
    if n_samples <= n_components:
        print(f"错误: 样本数 ({n_samples}) 不足以降维至 {n_components} 维。")
        return None, None
    try:
        tsne = TSNE(
            n_components=n_components,
            random_state=RANDOM_STATE,
            perplexity=effective_perplexity,
            max_iter=300,  # 可适当增加迭代次数以获得更好效果
            init="pca",  # 使用 PCA 初始化通常更快更稳定
            learning_rate="auto",  # 自动学习率
        )
        X_reduced = tsne.fit_transform(X)
        end_time = time.time()
        print(f"t-SNE 降维耗时: {end_time - start_time:.2f} 秒")
        return X_reduced, tsne
    except Exception as e:
        print(f"t-SNE 降维失败: {e}")
        return None, None


def plot_clusters_2d(
    X_reduced, labels, algorithm_name, dr_name, vis_method, n_clusters_found
):
    """
    在 2D 空间中绘制聚类结果并保存图像。
    (样式已还原)
    """
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(list(set(labels)))
    # 使用 viridis 颜色映射，与之前一致
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪声点用灰色、小标记 'x'
            col = [0.5, 0.5, 0.5, 1]  # 灰色
            marker = "x"
            markersize = 6  # 还原尺寸
            label = "Noise"
            alpha = 0.7  # 还原透明度
        else:
            # 正常簇点用彩色、标记 'o'
            marker = "o"
            markersize = 8  # 还原尺寸
            label = f"Cluster {k}"
            alpha = 0.7  # 还原透明度
        class_member_mask = labels == k
        xy = X_reduced[class_member_mask]
        # 注意：为避免 matplotlib 警告，将颜色包装在列表中
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            s=markersize,
            c=[col],
            marker=marker,
            label=label,
            alpha=alpha,
        )  # 移除了 edgecolors
    # 构建标题
    title = f"{algorithm_name} on {dr_name} Data (Vis: {vis_method})\nFound {n_clusters_found} clusters"
    plt.title(title)
    plt.xlabel(f"{vis_method} Component 1")
    plt.ylabel(f"{vis_method} Component 2")
    # 图例放回图内最佳位置
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.5)  # 保持网格样式
    plt.tight_layout()  # 使用 tight_layout 自动调整
    # 构建文件名
    base_filename = f"clusters_{clean_filename(algorithm_name)}_{dr_name}_vis{vis_method}_foundK{n_clusters_found}.png"
    save_path = os.path.join(SAVE_DIR, base_filename)
    plt.savefig(save_path)
    print(f"聚类结果图已保存至: {save_path}")
    plt.show()


def plot_unsupervised_comparison(results_list, dr_name, target_k):
    """
    绘制指定数据表示下不同聚类算法性能对比图 (轮廓系数和时间) 并保存。
    (样式已还原)
    """
    valid_results = [
        r
        for r in results_list
        if r.get("dr_name") == dr_name and r["score"] is not None and r["score"] != -1
    ]
    if not valid_results:
        print(f"在 {dr_name} 数据上没有有效的聚类结果可供比较。")
        return
    names = [r["name"] for r in valid_results]
    scores = [r["score"] for r in valid_results]
    times = [r["time"] for r in valid_results]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 绘制轮廓系数条形图 (还原样式)
    color_score = "tab:green"
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Silhouette Score", color=color_score)
    bars = ax1.bar(
        names, scores, color=color_score, alpha=0.6, label="Silhouette Score"
    )  # alpha=0.6
    ax1.tick_params(axis="y", labelcolor=color_score)
    # 动态调整 Y 轴范围 (保持动态调整，但确保包含 -0.2 到 0.5 的基础范围)
    min_s = min(scores) if scores else 0
    max_s = max(scores) if scores else 0
    ax1.set_ylim(min(min_s - 0.1, -0.2), max(max_s + 0.1, 0.5))
    # 在条形图上添加数值标签 (保持)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom" if yval >= 0 else "top",
            ha="center",
            fontsize=9,
        )
    # 创建第二个 Y 轴绘制处理时间 (还原样式)
    ax2 = ax1.twinx()
    color_time = "tab:red"
    ax2.set_ylabel("Processing Time (s)", color=color_time)
    # 使用 marker='o' 和 linestyle='--'
    ax2.plot(
        names,
        times,
        color=color_time,
        marker="o",
        linestyle="--",
        label="Processing Time",
    )
    ax2.tick_params(axis="y", labelcolor=color_time)
    max_t = max(times) if times else 1
    ax2.set_ylim(0, max_t * 1.1)
    # 添加标题和图例 (还原位置和方式)
    title = f"Clustering Algorithm Comparison on {dr_name} Data (Target K={target_k})"
    fig.suptitle(title)
    # 合并图例放在右上角
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # 将图例放在 ax1 的右上角变换坐标系中
    fig.legend(
        lines + lines2,
        labels + labels2,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        bbox_transform=ax1.transAxes,
    )
    plt.xticks(rotation=15)  # 还原旋转角度
    # 使用之前的 tight_layout 矩形调整
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # 构建文件名
    filename = f"comparison_{dr_name}_targetK{target_k}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    print(f"性能对比图已保存至: {save_path}")
    plt.show()


# --- 主函数 ---
def Unsupervised_Learning(
    data_processed,
    original_k,
    pca_k,
    tsne_k,
    n_components_pca=2,
    n_components_tsne=2,
    dbscan_params_original={"eps": 0.5, "min_samples": 5},
    dbscan_params_pca={"eps": 0.5, "min_samples": 5},
    dbscan_params_tsne={"eps": 0.5, "min_samples": 5},
    perplexity_tsne=30.0,
    standardize=True,  # 是否进行标准化
):
    """
    执行完整的无监督学习流程：
    1. 数据准备（可选标准化）
    2. 对原始数据、PCA降维数据、t-SNE降维数据分别进行处理：
       - 寻找 K 值
       - 运行 K-Means, Hierarchical, DBSCAN
       - 可视化聚类结果
       - 对比算法性能
    3. 返回所有聚类结果。
    """
    print("\n===================================")
    print("   开始执行无监督学习任务 (重构版)   ")
    print("===================================")
    # 1. 数据准备
    if not isinstance(data_processed, (pd.DataFrame, np.ndarray)):
        print("错误: 输入数据类型必须是 Pandas DataFrame 或 NumPy Array。")
        return []
    X_input = data_processed.copy()
    # 移除目标变量（如果存在）
    target_columns = ["GradeClass", "GPA"]
    if isinstance(X_input, pd.DataFrame):
        for col in target_columns:
            if col in X_input.columns:
                X_input = X_input.drop(col, axis=1)
                print(f"已移除 '{col}' 用于无监督学习。")
                break
        X_original = X_input.values  # 转换为 NumPy array
    else:  # 假设输入已经是 NumPy array
        X_original = X_input
        print("假设输入数据不含目标变量。")
    if X_original.size == 0 or X_original.shape[0] < 2:
        print("错误：用于无监督学习的数据为空或样本量过少！")
        return []
    print(f"原始数据维度: {X_original.shape}")
    # 可选：数据标准化 (强烈推荐用于距离敏感的算法如 K-Means, DBSCAN, PCA, t-SNE)
    if standardize:
        print("\n--- 标准化数据 (StandardScaler) ---")
        scaler = StandardScaler()
        X_original_scaled = scaler.fit_transform(X_original)
    else:
        X_original_scaled = X_original  # 不标准化
    all_results = []  # 存储所有聚类结果
    # --- 流程控制字典 ---
    # 定义三种数据表示的处理流程
    data_representations = {
        "Original": {
            "data": X_original_scaled,
            "target_k": original_k,
            "dbscan_params": dbscan_params_original,
            "needs_vis_reduction": True,  # 原始数据可视化需要降维
        },
        "PCA": {
            "data": None,  # 稍后填充
            "target_k": pca_k,
            "dbscan_params": dbscan_params_pca,
            "needs_vis_reduction": False,
        },
        "t-SNE": {
            "data": None,  # 稍后填充
            "target_k": tsne_k,
            "dbscan_params": dbscan_params_tsne,
            "needs_vis_reduction": False,
        },
    }
    # --- 执行降维 (PCA 和 t-SNE) ---
    # PCA
    X_pca, _, _ = apply_pca(X_original_scaled, n_components=n_components_pca)
    if X_pca is not None:
        data_representations["PCA"]["data"] = X_pca
    else:
        del data_representations["PCA"]  # 如果降维失败，则移除该流程
    # t-SNE
    X_tsne, _ = apply_tsne(
        X_original_scaled, n_components=n_components_tsne, perplexity=perplexity_tsne
    )
    if X_tsne is not None:
        data_representations["t-SNE"]["data"] = X_tsne
    else:
        if "t-SNE" in data_representations:
            del data_representations["t-SNE"]  # 如果降维失败，则移除该流程
    # --- 对每种数据表示执行聚类和评估 ---
    for dr_name, config in data_representations.items():
        print(f"\n{'='*15} 处理 {dr_name} 数据 {'='*15}")
        X_current = config["data"]
        target_k = config["target_k"]
        dbscan_params = config["dbscan_params"]
        needs_vis_reduction = config["needs_vis_reduction"]
        if X_current is None:
            print(f"{dr_name} 数据不可用，跳过处理。")
            continue
        # 1. 寻找 K 值
        find_optimal_k_elbow(
            X_current, max_k=MAX_K_SEARCH, target_k=target_k, dr_name=dr_name
        )
        current_results = []  # 存储当前数据表示下的结果
        # 2. 运行聚类算法
        # K-Means
        kmeans_res = run_kmeans(X_current, n_clusters=target_k, dr_name=dr_name)
        current_results.append(kmeans_res)
        # Hierarchical
        hierarchical_res = run_hierarchical(
            X_current, n_clusters=target_k, dr_name=dr_name
        )
        current_results.append(hierarchical_res)
        # DBSCAN
        dbscan_res = run_dbscan(X_current, params=dbscan_params, dr_name=dr_name)
        current_results.append(dbscan_res)
        # 添加到总结果列表
        all_results.extend(current_results)
        # 3. 可视化聚类结果
        print(f"\n--- 生成 {dr_name} 数据上的聚类可视化图表 ---")
        X_vis_pca = None
        X_vis_tsne = None
        # 如果需要为可视化进行降维 (仅针对原始数据)
        if needs_vis_reduction:
            print("为原始数据可视化进行临时降维...")
            # X_vis_pca, _, _ = apply_pca(X_current, n_components=2)
            X_vis_tsne, _ = apply_tsne(
                X_current, n_components=2, perplexity=perplexity_tsne
            )
        else:  # 如果数据本身就是降维后的 (PCA 或 t-SNE)
            if dr_name == "PCA":
                # X_vis_pca = X_current  # 直接使用 PCA 结果
                # 可以选择也用 t-SNE 可视化 PCA 结果
                X_vis_tsne, _ = apply_tsne(
                    X_current, n_components=2, perplexity=perplexity_tsne
                )
            elif dr_name == "t-SNE":
                X_vis_tsne = X_current  # 直接使用 t-SNE 结果
                # 可以选择也用 PCA 可视化 t-SNE 结果 (意义不大，但可以做)
                # X_vis_pca, _, _ = apply_pca(X_current, n_components=2)
        # 绘制每个算法的结果
        for result in current_results:
            algo_name = result["name"]
            labels = result["labels"]
            # DBSCAN 返回的是找到的簇数，KMeans/Hierarchical 返回目标簇数
            # 为了绘图标题一致，我们用实际标签中的簇数（不含噪声）
            unique_labels_plot = set(labels)
            n_clusters_plot = len(unique_labels_plot) - (
                1 if -1 in unique_labels_plot else 0
            )
            # 使用 PCA 进行可视化
            if X_vis_pca is not None:
                plot_clusters_2d(
                    X_vis_pca, labels, algo_name, dr_name, "PCA", n_clusters_plot
                )
            # 使用 t-SNE 进行可视化
            if X_vis_tsne is not None:
                plot_clusters_2d(
                    X_vis_tsne, labels, algo_name, dr_name, "t-SNE", n_clusters_plot
                )
        # 4. 绘制性能对比图
        print(f"\n--- 生成 {dr_name} 数据上的算法性能对比图 ---")
        plot_unsupervised_comparison(
            current_results, dr_name=dr_name, target_k=target_k
        )
    print("\n===================================")
    print("   无监督学习任务执行完毕   ")
    print("===================================")
    return all_results


if __name__ == "__main__":
    data = description_wash()
    data.to_csv(WASHED_DATA_PATH, index=False, encoding="utf-8")
    categorical_features_to_encode = [
        "Ethnicity",
        "ParentalEducation",
        "ParentalInvolvement",
    ]

    data_processed = preprocess_data(data, categorical_features_to_encode)
    data1 = data_processed.copy()
    data_processed = data_processed.drop("Age", axis=1)
    data_processed = data_processed.drop("Gender", axis=1)
    columns_to_drop = [col for col in data_processed.columns if "Ethnicity" in col]
    data_processed = data_processed.drop(columns=columns_to_drop, axis=1)
    data_processed = data_processed.drop("Extracurricular", axis=1)
    data_processed = data_processed.drop("Sports", axis=1)
    data_processed = data_processed.drop("Music", axis=1)
    data_processed = data_processed.drop("Volunteering", axis=1)
    data_processed.to_csv(PROCESSED_DATA_PATH, index=False, encoding="utf-8")

    # supervised_results = Supervised_Learning(data_processed)
    # unsupervised_results = Unsupervised_Learning(data_processed,
    #                                              pca_k=12,
    #                                              tsne_k=24,
    #                                              n_components_pca=2, # PCA 降到 2 维
    #                                              n_components_tsne=2) # t-SNE 降到 2 维

    # supervised_results = Supervised_Learning(data1)
    # unsupervised_results = Unsupervised_Learning(data1, 2)
    k_for_original = 10
    k_for_pca = 4
    k_for_tsne = 24  # t-SNE 上的 K 值可能与其他不同
    # 2. 定义 DBSCAN 参数 (需要根据数据仔细调整！)
    # 这些只是示例值，实际效果依赖数据分布
    dbscan_params_orig = {"eps": 1.75, "min_samples": 25}  # 原始标准化数据
    dbscan_params_pca_2d = {"eps": 0.5, "min_samples": 3}  # 2D PCA 数据
    dbscan_params_tsne_2d = {"eps": 0.5, "min_samples": 3}  # 2D t-SNE 数据
    # 3. 运行主函数
    all_clustering_results = Unsupervised_Learning(
        data_processed=data_processed,  # 你的数据
        original_k=k_for_original,
        pca_k=k_for_pca,
        tsne_k=k_for_tsne,
        n_components_pca=2,  # PCA 降维到 2 维
        n_components_tsne=2,  # t-SNE 降维到 2 维
        dbscan_params_original=dbscan_params_orig,
        dbscan_params_pca=dbscan_params_pca_2d,
        dbscan_params_tsne=dbscan_params_tsne_2d,
        perplexity_tsne=30.0,  # t-SNE 的 perplexity 参数
    )
    # 4. 查看结果总结 (可选)
    print("\n--- 所有聚类结果总结 ---")
    if all_clustering_results:
        summary_df = pd.DataFrame(all_clustering_results)
        # 格式化输出
        summary_df["score"] = summary_df["score"].map("{:.4f}".format)
        summary_df["time"] = summary_df["time"].map("{:.4f}s".format)
        print(summary_df[["dr_name", "name", "n_clusters", "score", "time"]])
    else:
        print("未能生成任何聚类结果。")

    print("\n所有流程执行完毕。")
