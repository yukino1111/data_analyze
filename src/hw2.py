from config import *
from common import description_wash

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd  # 假设 description_wash() 返回 Pandas DataFrame

# --- 监督学习所需库 ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC  # "Support Vector Classification"
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # 用于获取类别标签
import time  # 用于计算训练时间

# --- Unsupervised Learning Imports ---
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
)  # For potential dendrogram plotting (optional)
import warnings  # To ignore specific warnings like KMeans future changes

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.cluster._kmeans"
)
from sklearn.exceptions import UndefinedMetricWarning  # 导入特定的警告类型

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def preprocess_data(df, categorical_cols_to_encode):
    """对数值列标准化，对指定分类列进行独热编码"""
    df_processed = df.copy()
    numerical_features1 = ["GPA"]
    numerical_features2 = ["StudyTimeWeekly", "Absences","Age"]
    if numerical_features1:  # 检查列表是否为空
        scaler = StandardScaler()
        df_processed[numerical_features1] = scaler.fit_transform(
            df[numerical_features1]
        )
        print(f"数值特征 {numerical_features1} 已进行标准化处理。")
    if numerical_features2:  # 检查列表是否为空
        scaler = MinMaxScaler()
        df_processed[numerical_features2] = scaler.fit_transform(
            df[numerical_features2]
        )
        print(f"数值特征 {numerical_features2} 已进行归一化处理 (Min-Max 缩放)。")
    # 2. 独热编码分类特征
    if categorical_cols_to_encode:  # 检查列表是否为空
        # 使用 get_dummies 进行独热编码
        # drop_first=True 可以减少一个维度，避免多重共线性，常用于线性模型
        # df_processed = pd.get_dummies(
        #     df_processed, columns=categorical_cols_to_encode,  dtype=int
        # )
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
    report = classification_report(
        y_test, y_pred, target_names=grade_labels
    )  # 使用后面定义的 grade_labels
    cm = confusion_matrix(y_test, y_pred)
    training_time = end_time - start_time

    print(f"训练时间: {training_time:.4f} 秒")
    print(f"准确率: {accuracy:.4f}")
    # print("分类报告:\n", report) # 报告可能很长，先注释掉，需要时再打开
    # print("混淆矩阵:\n", cm)

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
    # print("分类报告:\n", report)
    # print("混淆矩阵:\n", cm)

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

    # 在条形图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom",
            ha="center",
        )  # va='bottom' 放在条形图上方

    # 创建第二个 Y 轴共享 X 轴，用于绘制训练时间
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
    plt.show()


# 定义 GradeClass 标签映射 (用于报告和绘图) - 确保这个定义在调用它的函数之前
grade_labels_map = {
    0: "A (>=3.5)",
    1: "B (3.0-3.5)",
    2: "C (2.5-3.0)",
    3: "D (2.0-2.5)",
    4: "F (<2.0)",
}
# 稍后我们会根据 y 中的实际值动态生成 grade_labels
grade_labels = []  # 先初始化为空列表


# --- 3. 修改主函数流程 ---
def Supervised_Learning(data_processed1):
    """执行监督学习流程：数据分割、模型训练、评估和可视化"""
    print("\n==========================")
    print("   开始执行监督学习任务   ")
    print("==========================")
    data_processed = data_processed1.copy()
    # data_processed = data.drop("GPA", axis=1)
    # 1. 分离特征 (X) 和目标变量 (y)
    if "GradeClass" not in data_processed.columns:
        print("错误：目标变量 'GradeClass' 不在处理后的数据中！")
        return

    X = data_processed.drop("GradeClass", axis=1)
    y = data_processed["GradeClass"]

    # 动态生成 grade_labels，确保顺序与 y 中的唯一值一致
    global grade_labels  # 声明我们要修改全局变量 grade_labels
    unique_grades = sorted(y.unique())  # 获取排序后的唯一等级值
    grade_labels = [
        grade_labels_map.get(g, f"Unknown({g})") for g in unique_grades
    ]  # 使用映射生成标签
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
    results = []  # 用于存储每个模型的结果

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

    return results  # 可以选择返回结果供后续分析


def find_optimal_k_elbow(X, max_k=11, random_state=42):
    """使用肘部法则寻找 K-Means 的最佳 K 值并绘图，同时计算轮廓系数和 Calinski-Harabasz 指数"""
    print("\n--- 寻找 K-Means 的最佳 K (肘部法则) ---")
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
    plt.figure(figsize=(18, 5))  # 增加图像宽度，方便显示多个子图
    plt.subplot(1, 3, 1)  # 1 行 3 列，选择第 1 个子图
    plt.plot(k_range, inertias, marker="o")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia (WCSS)")
    plt.xticks(k_range)
    plt.grid(True)
    # 绘制轮廓系数图
    plt.subplot(1, 3, 2)  # 1 行 3 列，选择第 2 个子图
    plt.plot(k_range, silhouette_scores, marker="o")
    plt.title("Silhouette Score for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_range)
    plt.grid(True)
    # 绘制 Calinski-Harabasz 指数图
    plt.subplot(1, 3, 3)  # 1 行 3 列，选择第 3 个子图
    plt.plot(k_range, calinski_harabasz_scores, marker="o")
    plt.title("Calinski-Harabasz Index for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Calinski-Harabasz Index")
    plt.xticks(k_range)
    plt.grid(True)
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.show()
    print(
        "请观察上面的肘部图、轮廓系数图和 Calinski-Harabasz 指数图，综合确定一个合适的 K 值。"
    )
    return k_range, inertias, silhouette_scores, calinski_harabasz_scores


def run_kmeans(X, n_clusters, random_state=42):
    """运行 K-Means 聚类"""
    print(f"\n--- 运行 K-Means (K={n_clusters}) ---")
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
        "name": "K-Means",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters,
    }


def run_hierarchical(X, n_clusters):
    """运行层次聚类 (Agglomerative)"""
    print(f"\n--- 运行层次聚类 (n_clusters={n_clusters}) ---")
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
        "name": "Hierarchical (Ward)",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters,
    }


def run_dbscan(X, eps=0.5, min_samples=5):
    """运行 DBSCAN 聚类"""
    print(f"\n--- 运行 DBSCAN (eps={eps}, min_samples={min_samples}) ---")
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
        "name": f"DBSCAN (eps={eps}, min={min_samples})",
        "labels": labels,
        "score": score,
        "time": training_time,
        "n_clusters": n_clusters_found,
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
        n_iter=300,
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
        f"{algorithm_name} Clusters (Projected by {dr_name})\nFound {n_clusters_found} clusters (excl. noise)"
    )
    plt.xlabel(f"{dr_name} Component 1")
    plt.ylabel(f"{dr_name} Component 2")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()


def plot_unsupervised_comparison(results):
    """绘制不同聚类算法性能对比图 (轮廓系数和时间)"""
    # 过滤掉 score 为 -1 (计算失败) 的结果，避免绘图错误
    valid_results = [r for r in results if r["score"] != -1 and r["score"] is not None]
    if not valid_results:
        print("没有有效的轮廓系数可供比较。")
        return

    names = [r["name"] for r in valid_results]
    scores = [r["score"] for r in valid_results]
    times = [r["time"] for r in valid_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制轮廓系数条形图
    color = "tab:green"
    ax1.set_xlabel("Algorithm")
    ax1.set_ylabel("Silhouette Score", color=color)
    bars = ax1.bar(names, scores, color=color, alpha=0.6, label="Silhouette Score")
    ax1.tick_params(axis="y", labelcolor=color)
    # 调整 Y 轴范围，轮廓系数在 [-1, 1] 之间
    ax1.set_ylim(
        min(min(scores) - 0.1, -0.2), max(max(scores) + 0.1, 0.5)
    )  # 动态调整，但至少包含部分负数区域

    # 在条形图上显示数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom" if yval >= 0 else "top",
            ha="center",
        )

    # 创建第二个 Y 轴共享 X 轴，用于绘制处理时间
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Processing Time (s)", color=color)
    ax2.plot(
        names, times, color=color, marker="o", linestyle="--", label="Processing Time"
    )
    ax2.tick_params(axis="y", labelcolor=color)
    # 根据时间调整 Y 轴范围，确保从 0 开始
    ax2.set_ylim(0, max(times) * 1.1)

    fig.suptitle("Clustering Algorithm Comparison: Silhouette Score and Time")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.xticks(rotation=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def Unsupervised_Learning(data_processed):
    """执行无监督学习流程：聚类、降维和评估"""
    print("\n=============================")
    print("   开始执行无监督学习任务   ")
    print("=============================")

    # 1. 准备数据：通常不使用目标变量
    if "GradeClass" in data_processed.columns:
        X_unsupervised = data_processed.drop("GradeClass", axis=1)
        print("已移除 'GradeClass' 用于无监督学习。")
    else:
        X_unsupervised = data_processed.copy()
        print("未找到 'GradeClass'，使用所有提供的特征进行无监督学习。")

    if X_unsupervised.empty:
        print("错误：用于无监督学习的数据为空！")
        return

    print(f"用于聚类的数据维度: {X_unsupervised.shape}")

    # 2. K-Means: 先用肘部法则确定 K
    find_optimal_k_elbow(X_unsupervised, max_k=20)
    # *** 手动选择 K ***
    # 基于上面的肘部图，或者你的领域知识，选择一个 K 值
    # 例如，如果图表在 K=4 或 K=5 处有拐点，或者你知道有 5 个等级
    chosen_k = 17  # <--- 在这里设置你选择的 K 值
    print(f"\n*** 基于肘部图或先验知识，选择 K = {chosen_k} ***")

    results = []  # 存储结果

    # 运行 K-Means
    kmeans_results = run_kmeans(X_unsupervised, n_clusters=chosen_k)
    results.append(kmeans_results)

    # 3. 运行层次聚类 (使用与 K-Means 相同的 K 以便比较)
    hierarchical_results = run_hierarchical(X_unsupervised, n_clusters=chosen_k)
    results.append(hierarchical_results)

    # 4. 运行 DBSCAN
    # DBSCAN 的参数 eps 和 min_samples 对结果影响很大，且依赖数据
    # 这里的 0.5 和 5 是示例值，对于标准化后的数据可能是一个起点
    # 你可能需要根据数据特性和轮廓系数反馈来调整这些值
    # 例如，如果噪声点太多，尝试增大 eps 或 min_samples
    # 如果簇太少或太大，尝试减小 eps
    dbscan_eps = 1.25  # <--- 示例值，需要调整
    dbscan_min_samples = 17  # <--- 示例值，需要调整
    dbscan_results = run_dbscan(
        X_unsupervised, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    results.append(dbscan_results)

    # 5. 降维用于可视化
    # 应用 PCA
    X_pca, pca_model = apply_pca(X_unsupervised)
    # 应用 t-SNE (如果数据量大，这步会比较慢)
    X_tsne, tsne_model = apply_tsne(X_unsupervised)  # 使用默认 perplexity

    # 6. 可视化聚类结果 (使用 PCA 和 t-SNE 降维后的数据)
    print("\n--- 生成聚类结果可视化图表 ---")
    if X_pca is not None:
        for result in results:
            plot_clusters_2d(
                X_pca, result["labels"], result["name"], "PCA", result["n_clusters"]
            )

    if X_tsne is not None:
        for result in results:
            # DBSCAN 可能找到的簇数与 K-Means/Hierarchical 不同
            plot_clusters_2d(
                X_tsne, result["labels"], result["name"], "t-SNE", result["n_clusters"]
            )
    else:
        print("t-SNE 降维失败，跳过 t-SNE 可视化。")

    # 7. 绘制性能对比图
    print("\n--- 生成算法性能对比图 ---")
    plot_unsupervised_comparison(results)

    print("\n=============================")
    print("   无监督学习任务执行完毕   ")
    print("=============================")

    return results  # 返回结果供后续分析


if __name__ == "__main__":
    data = description_wash()
    categorical_features_to_encode = [
        "Ethnicity",
        "ParentalEducation",
        "ParentalInvolvement",
    ]

    # 注意：二元特征和 GradeClass 不在这里
    data_processed = preprocess_data(data, categorical_features_to_encode)

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
    unsupervised_results = Unsupervised_Learning(data_processed)
    print("\n所有流程执行完毕。")
