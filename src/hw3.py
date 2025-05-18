import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import community as community_louvain
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from collections import Counter


# 设置字体路径
font_path = "assets/fonts/PingFang-Medium.ttf"
font_prop = fm.FontProperties(fname=font_path)

# 创建保存图片的目录
import os

if not os.path.exists("assets/pic"):
    os.makedirs("assets/pic")


class SocialNetworkAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.graph = self._load_graph()

    def _load_graph(self):
        """加载图数据"""
        G = nx.read_edgelist(self.file_path, create_using=nx.Graph())
        return G

    def _save_plot(self, fig, filename):
        """保存图表"""
        fig.savefig(f"assets/pic/{filename}.png", bbox_inches="tight")
        plt.close(fig)

    def analyze_node_importance(self):
        """分析节点重要性"""
        print("--- 节点重要性分析 ---")

        # 度中心性
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_degree = sorted(
            degree_centrality.items(), key=lambda item: item[1], reverse=True
        )
        print("度中心性 Top 10:", sorted_degree[:10])
        self._plot_top_nodes(
            degree_centrality, "度中心性 Top 10", "degree_centrality_top10"
        )

        # 介数中心性
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        sorted_betweenness = sorted(
            betweenness_centrality.items(), key=lambda item: item[1], reverse=True
        )
        print("介数中心性 Top 10:", sorted_betweenness[:10])
        self._plot_top_nodes(
            betweenness_centrality, "介数中心性 Top 10", "betweenness_centrality_top10"
        )

        # 特征向量中心性
        eigenvector_centrality = nx.eigenvector_centrality(self.graph)
        sorted_eigenvector = sorted(
            eigenvector_centrality.items(), key=lambda item: item[1], reverse=True
        )
        print("特征向量中心性 Top 10:", sorted_eigenvector[:10])
        self._plot_top_nodes(
            eigenvector_centrality,
            "特征向量中心性 Top 10",
            "eigenvector_centrality_top10",
        )

        # 接近中心性
        closeness_centrality = nx.closeness_centrality(self.graph)
        sorted_closeness = sorted(
            closeness_centrality.items(), key=lambda item: item[1], reverse=True
        )
        print("接近中心性 Top 10:", sorted_closeness[:10])
        self._plot_top_nodes(
            closeness_centrality, "接近中心性 Top 10", "closeness_centrality_top10"
        )

        # PageRank
        pagerank = nx.pagerank(self.graph)
        sorted_pagerank = sorted(
            pagerank.items(), key=lambda item: item[1], reverse=True
        )
        print("PageRank Top 10:", sorted_pagerank[:10])
        self._plot_top_nodes(pagerank, "PageRank Top 10", "pagerank_top10")

    def analyze_edge_importance(self):
        """分析边重要性"""
        print("\n--- 边重要性分析 ---")

        # 边介数
        edge_betweenness = nx.edge_betweenness_centrality(self.graph)
        sorted_edge_betweenness = sorted(
            edge_betweenness.items(), key=lambda item: item[1], reverse=True
        )
        print("边介数 Top 10:", sorted_edge_betweenness[:10])

        # 绘制边介数 Top 10 图
        self._plot_top_edges(
            edge_betweenness, "边介数 Top 10", "edge_betweenness_top10"
        )

    def _plot_top_edges(self, edge_centrality_dict, title, filename):
        """绘制边中心性 Top 10 边图"""
        top_edges = sorted(
            edge_centrality_dict.items(), key=lambda item: item[1], reverse=True
        )[:10]
        edges, values = zip(*top_edges)

        # 将边表示为字符串，例如 "节点1-节点2"
        edge_labels = [f"{u}-{v}" for u, v in edges]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(edge_labels, values)
        ax.set_title(title, fontproperties=font_prop)
        ax.set_xlabel("边", fontproperties=font_prop)
        ax.set_ylabel("中心性值", fontproperties=font_prop)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save_plot(fig, filename)

    def _plot_top_nodes(self, centrality_dict, title, filename):
        """绘制中心性 Top 10 节点图"""
        top_nodes = sorted(
            centrality_dict.items(), key=lambda item: item[1], reverse=True
        )[:10]
        nodes, values = zip(*top_nodes)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(nodes, values)
        ax.set_title(title, fontproperties=font_prop)
        ax.set_xlabel("节点", fontproperties=font_prop)
        ax.set_ylabel("中心性值", fontproperties=font_prop)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save_plot(fig, filename)

    def find_communities(self, algorithm="louvain"):
        """社团发现"""
        print(f"\n--- 社团发现 ({algorithm}) ---")
        partition = None
        modularity = None
        communities_list = []  # 用于存储社区列表，方便计算其他指标

        if algorithm == "louvain":
            partition = community_louvain.best_partition(self.graph)
            modularity = community_louvain.modularity(partition, self.graph)
            # 从 partition 构建 communities_list
            communities_dict = defaultdict(list)
            for node, comm_id in partition.items():
                communities_dict[comm_id].append(node)
            communities_list = list(communities_dict.values())

        elif algorithm == "newman":
            communities_list = list(
                nx.community.greedy_modularity_communities(self.graph)
            )
            partition = {}
            for i, comm in enumerate(communities_list):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(self.graph, communities_list)

        elif algorithm == "label_propagation":
            communities_list = list(
                nx.community.label_propagation_communities(self.graph)
            )
            partition = {}
            for i, comm in enumerate(communities_list):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(self.graph, communities_list)

        elif algorithm == "hierarchical":
            adj_matrix = nx.to_numpy_array(self.graph)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity="euclidean",
                linkage="ward",
                distance_threshold=0.5,
            )
            clustering.fit(adj_matrix)
            partition = {
                node: label
                for node, label in zip(self.graph.nodes(), clustering.labels_)
            }
            communities_dict = defaultdict(list)
            for node, comm_id in partition.items():
                communities_dict[comm_id].append(node)
            communities_list = list(communities_dict.values())
            modularity = nx.community.modularity(self.graph, communities_list)

        else:
            print("未知社团发现算法")
            return None, None

        if partition is None:
            print("未能发现社团。")
            return None, None

        # 计算并打印其他参数
        community_count = len(communities_list)
        print(f"社团数量: {community_count}")

        if community_count > 0:
            community_sizes_list = [len(comm) for comm in communities_list]
            avg_community_size = np.mean(community_sizes_list)
            print(f"平均社团规模: {avg_community_size:.4f}")

            print(f"模块度: {modularity:.4f}")  # 打印模块度

            # 计算平均内部密度和平均电导率
            internal_densities = []
            conductances = []
            for comm_nodes in communities_list:
                if len(comm_nodes) > 1:  # 内部密度和电导率对单人社团没有意义
                    subgraph = self.graph.subgraph(comm_nodes)
                    internal_densities.append(nx.density(subgraph))
                    conductances.append(nx.conductance(self.graph, comm_nodes))

            if internal_densities:
                avg_internal_density = np.mean(internal_densities)
                print(f"平均内部密度 (非单人社团): {avg_internal_density:.4f}")
            else:
                print("没有非单人社团，无法计算平均内部密度。")

            if conductances:
                avg_conductance = np.mean(conductances)
                print(f"平均电导率 (非单人社团): {avg_conductance:.4f}")
            else:
                print("没有非单人社团，无法计算平均电导率。")

        # 社团规模分布
        community_sizes = defaultdict(int)
        for comm_id in partition.values():
            community_sizes[comm_id] += 1
        sorted_community_sizes = sorted(
            community_sizes.items(), key=lambda item: item[1], reverse=True
        )
        # 过滤出规模大于等于2的社团的规模
        communities_size_ge2 = [
            size for community_id, size in sorted_community_sizes if size >= 2
        ]
        print("社团规模分布 (Top 10):", sorted_community_sizes[:10])
        self._plot_community_size_distribution(
            sorted_community_sizes,
            f"{algorithm} 社团规模分布 (Top 10)",
            f"{algorithm}_community_size_distribution_top10",
        )
        # 构建社团ID到节点列表的映射 (如果之前没有构建过的话)
        # 如果你在其他地方已经构建了 community_nodes_map，可以跳过这部分
        community_nodes_map = defaultdict(list)
        for node, comm_id in partition.items():
            community_nodes_map[comm_id].append(node)
        # 过滤出规模大于等于2的社团的ID
        communities_id_ge2 = [
            community_id
            for community_id, nodes in community_nodes_map.items()
            if len(nodes) >= 2
        ]

        # 社团关系概览 (规模 >= 2)
        self._plot_community_relationship_overview(
            partition,
            communities_id_ge2,  # 传递过滤后的社团ID列表
            f"{algorithm} 社团关系概览 (规模 >= 2)",
            f"{algorithm}_community_relationship_overview_ge2",
        )

        # 全部社团连接密度
        self._print_community_density(partition, f"{algorithm} 全部社团连接密度")

        # 绘制其他社团分析图表 (针对最好的算法)
        # 这里我们假设louvain是最好的算法，或者你可以根据模块度选择
        if algorithm == "louvain":  # 或者根据模块度判断
            self._plot_community_analysis(partition, algorithm)

        return partition, modularity

    def _plot_community_size_distribution(
        self, sorted_community_sizes, title, filename
    ):
        """绘制社团规模分布图"""
        community_ids, sizes = zip(*sorted_community_sizes)
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar([str(c) for c in community_ids[:10]], sizes[:10])
        ax.set_title(title, fontproperties=font_prop)
        ax.set_xlabel("社团ID", fontproperties=font_prop)
        ax.set_ylabel("成员数量", fontproperties=font_prop)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save_plot(fig, filename)

    def _plot_community_relationship_overview(
        self, partition, community_ids_to_plot, title, filename
    ):
        """绘制社团关系概览图和社团间连接密度热力图 (针对指定社团ID列表)"""
        # print(f"绘制 {title}")

        # 传入的参数已经是需要绘制的社团ID列表
        top_community_ids = community_ids_to_plot
        num_top_communities = len(top_community_ids)

        if num_top_communities == 0:
            print("没有社团可供绘制。")
            return

        # 创建一个表示社团之间连接的图
        community_graph = nx.Graph()

        # 构建社团ID到节点列表的映射，并计算社团规模
        community_nodes_map = defaultdict(list)
        community_sizes = {}
        for node, comm_id in partition.items():
            if comm_id in top_community_ids:  # 只处理需要绘制的社团
                community_nodes_map[comm_id].append(node)

        for comm_id in top_community_ids:
            size = len(
                community_nodes_map.get(comm_id, [])
            )  # 获取社团规模，如果社团ID不存在则规模为0
            community_sizes[comm_id] = size
            community_graph.add_node(comm_id, size=size)

        # 计算社团之间的连接数量和密度矩阵
        # 初始化连接数量矩阵和密度矩阵
        connection_matrix = np.zeros((num_top_communities, num_top_communities))
        density_matrix = np.zeros((num_top_communities, num_top_communities))
        community_id_to_index = {
            comm_id: i for i, comm_id in enumerate(top_community_ids)
        }

        for i in range(num_top_communities):
            for j in range(i + 1, num_top_communities):
                comm_id1 = top_community_ids[i]
                comm_id2 = top_community_ids[j]
                comm1_nodes = community_nodes_map.get(
                    comm_id1, []
                )  # 使用get避免KeyError
                comm2_nodes = community_nodes_map.get(comm_id2, [])

                edges = sum(
                    1
                    for node1 in comm1_nodes
                    for node2 in comm2_nodes
                    if self.graph.has_edge(node1, node2)
                )

                connection_matrix[i, j] = connection_matrix[j, i] = edges

                # 计算社团间连接密度
                # 可能的最大连接数是 len(comm1_nodes) * len(comm2_nodes)
                max_possible_edges = len(comm1_nodes) * len(comm2_nodes)
                if max_possible_edges > 0:
                    density = edges / max_possible_edges
                    density_matrix[i, j] = density_matrix[j, i] = density

                if edges > 0:
                    community_graph.add_edge(comm_id1, comm_id2, weight=edges)

        # 绘制社团关系概览图
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(community_graph)

        # 获取节点大小和边权重
        sizes = [
            community_graph.nodes[node]["size"] * 10 for node in community_graph.nodes()
        ]  # 调整乘数以获得合适的大小
        edge_weights = [
            community_graph.edges[edge]["weight"] * 0.1
            for edge in community_graph.edges()
        ]  # 调整乘数以获得合适的宽度

        # 绘制节点
        nx.draw_networkx_nodes(
            community_graph,
            pos,
            node_size=sizes,
            node_color=range(num_top_communities),  # 使用社团的数量作为颜色范围
            cmap=plt.cm.rainbow,
            ax=ax1,
        )

        # 绘制边
        nx.draw_networkx_edges(
            community_graph, pos, width=edge_weights, alpha=0.5, ax=ax1
        )

        # 绘制标签
        labels = {
            comm_id: f'社团{comm_id}\n({community_graph.nodes[comm_id]["size"]}人)'
            for comm_id in top_community_ids
        }
        nx.draw_networkx_labels(
            community_graph,
            pos,
            labels=labels,
            font_family=font_prop.get_name(),
            ax=ax1,
        )

        ax1.set_title(title, fontproperties=font_prop)
        ax1.axis("off")
        plt.tight_layout()
        self._save_plot(fig1, filename)

        # 绘制社团间连接密度热力图
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        im = ax2.imshow(density_matrix, cmap="YlOrRd", interpolation="nearest")

        # 添加颜色条
        cbar = ax2.figure.colorbar(im, ax=ax2)
        cbar.ax.set_ylabel(
            "连接密度", rotation=-90, va="bottom", fontproperties=font_prop
        )

        # 设置刻度标签
        ax2.set_xticks(np.arange(num_top_communities))
        ax2.set_yticks(np.arange(num_top_communities))
        ax2.set_xticklabels(
            [f"社团{comm_id}" for comm_id in top_community_ids],
            fontproperties=font_prop,
        )
        ax2.set_yticklabels(
            [f"社团{comm_id}" for comm_id in top_community_ids],
            fontproperties=font_prop,
        )

        # 旋转 x 轴标签
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 添加密度值到热力图
        for i in range(num_top_communities):
            for j in range(num_top_communities):
                text = ax2.text(
                    j,
                    i,
                    f"{density_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="w" if density_matrix[i, j] < 0.5 else "black",
                )

        ax2.set_title(f"{title} - 连接密度热力图", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(fig2, f"{filename}_heatmap")

    def _print_community_density(self, partition, title):
        """打印全部社团连接密度"""
        print(f"\n{title}:")
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        zero_density_communities = []
        for comm_id, nodes in communities.items():
            subgraph = self.graph.subgraph(nodes)
            density = nx.density(subgraph)
            if density == 0:
                zero_density_communities.append(comm_id)
            else:
                print(f"  社团 {comm_id} 密度: {density:.4f}")
        if zero_density_communities:
            print(f"  密度为 0 的社团: {', '.join(map(str, zero_density_communities))}")

    def _plot_community_analysis(self, partition, algorithm):
        """绘制社团分析相关的图表"""
        # print(f"\n--- 社团分析图表 ({algorithm}) ---")

        # 节点度中心性 (Top 10)
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_degree = sorted(
            degree_centrality.items(), key=lambda item: item[1], reverse=True
        )
        top_10_nodes = [node for node, centrality in sorted_degree[:10]]
        top_10_centralities = [centrality for node, centrality in sorted_degree[:10]]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(top_10_nodes, top_10_centralities)
        ax.set_title(f"{algorithm} 节点度中心性 (Top 10)", fontproperties=font_prop)
        ax.set_xlabel("节点", fontproperties=font_prop)
        ax.set_ylabel("度中心性值", fontproperties=font_prop)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self._save_plot(fig, f"{algorithm}_node_degree_centrality_top10")

        # 大小节点度分布
        degree_sequence = sorted([d for n, d in self.graph.degree()], reverse=True)
        degree_counts = defaultdict(int)
        for degree in degree_sequence:
            degree_counts[degree] += 1
        degrees, counts = zip(*sorted(degree_counts.items()))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(degrees, counts, "o-")
        ax.set_title(f"{algorithm} 节点度分布", fontproperties=font_prop)
        ax.set_xlabel("度", fontproperties=font_prop)
        ax.set_ylabel("节点数量", fontproperties=font_prop)
        plt.grid(True)
        self._save_plot(fig, f"{algorithm}_degree_distribution")

        # 聚类系数分布
        clustering_coefficients = nx.clustering(self.graph)
        coeff_values = list(clustering_coefficients.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(coeff_values, bins=20)
        ax.set_title(f"{algorithm} 聚类系数分布", fontproperties=font_prop)
        ax.set_xlabel("聚类系数", fontproperties=font_prop)
        ax.set_ylabel("节点数量", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(fig, f"{algorithm}_clustering_coefficient_distribution")

        # 度-聚类分布
        avg_clustering_per_degree = nx.average_clustering(
            self.graph, nodes=self.graph.nodes(), weight=None
        )
        print(f"{algorithm} 平均聚类系数: {avg_clustering_per_degree:.4f}")
        # 绘制度-聚类分布图需要更复杂的数据处理，这里简化为打印平均值

        # 社团内部连接密度热力图 (Top 10 社团)
        # print(f"\n绘制 {algorithm} 社团内部连接密度热力图 (Top 10 社团)")
        community_sizes = defaultdict(list)
        for node, comm_id in partition.items():
            community_sizes[comm_id].append(node)

        sorted_communities = sorted(
            community_sizes.items(), key=lambda item: len(item[1]), reverse=True
        )
        top_10_communities = sorted_communities[:10]
        top_10_community_ids = [comm_id for comm_id, nodes in top_10_communities]
        num_top_10_communities = len(top_10_community_ids)

        if num_top_10_communities == 0:
            print("没有 Top 10 社团可供绘制内部密度热力图。")
            return

        internal_density_matrix = np.zeros(
            (num_top_10_communities, num_top_10_communities)
        )
        community_id_to_index = {
            comm_id: i for i, comm_id in enumerate(top_10_community_ids)
        }

        for i in range(num_top_10_communities):
            comm_id = top_10_community_ids[i]
            nodes = community_sizes[comm_id]
            subgraph = self.graph.subgraph(nodes)
            density = nx.density(subgraph)
            internal_density_matrix[i, i] = density  # 对角线表示社团内部密度

        fig3, ax3 = plt.subplots(figsize=(10, 8))
        im = ax3.imshow(
            internal_density_matrix, cmap="Reds", interpolation="nearest"
        )  # 使用 Reds 颜色映射

        # 添加颜色条
        cbar = ax3.figure.colorbar(im, ax=ax3)
        cbar.ax.set_ylabel(
            "内部连接密度", rotation=-90, va="bottom", fontproperties=font_prop
        )

        # 设置刻度标签
        ax3.set_xticks(np.arange(num_top_10_communities))
        ax3.set_yticks(np.arange(num_top_10_communities))
        ax3.set_xticklabels(
            [f"社团{comm_id}" for comm_id in top_10_community_ids],
            fontproperties=font_prop,
        )
        ax3.set_yticklabels(
            [f"社团{comm_id}" for comm_id in top_10_community_ids],
            fontproperties=font_prop,
        )

        # 旋转 x 轴标签
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 添加密度值到热力图
        for i in range(num_top_10_communities):
            # 只在对角线上添加文本
            text = ax3.text(
                i,
                i,
                f"{internal_density_matrix[i, i]:.2f}",
                ha="center",
                va="center",
                color="w" if internal_density_matrix[i, i] < 0.5 else "black",
            )

        ax3.set_title(
            f"{algorithm} 社团内部连接密度热力图 (Top 10 社团)",
            fontproperties=font_prop,
        )
        plt.tight_layout()
        self._save_plot(fig3, f"{algorithm}_internal_density_heatmap_top10")

    def compare_algorithms(self):
        """比较不同社团发现算法的模块度"""
        print("\n--- 社团发现算法模块度对比 ---")
        algorithms = ["louvain", "newman", "label_propagation"]
        modularity_scores = {}
        community_counts = {}  # 存储不同算法下的社团数量

        for algo in algorithms:
            print(f"运行 {algo} 算法...")
            partition, modularity = self.find_communities(algorithm=algo)
            if modularity is not None:
                modularity_scores[algo] = modularity
            if partition is not None:
                # 计算社团数量
                num_communities = len(set(partition.values()))
                community_counts[algo] = num_communities
                print(f"  发现社团数量: {num_communities}")

        print("\n模块度对比结果:")
        for algo, mod in modularity_scores.items():
            print(f"  {algo}: {mod:.4f}")
        print("\n社团数量对比结果:")
        for algo, count in community_counts.items():
            print(f"  {algo}: {count}")
        # 绘制模块度对比图
        fig, ax = plt.subplots(figsize=(8, 5))
        algorithms = list(modularity_scores.keys())
        modularity_values = list(modularity_scores.values())
        ax.bar(algorithms, modularity_values)
        ax.set_title("社团发现算法模块度对比", fontproperties=font_prop)
        ax.set_xlabel("算法", fontproperties=font_prop)
        ax.set_ylabel("模块度", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(fig, "modularity_comparison")
        # 绘制社团数量对比图 (可选，如果需要可以添加)
        fig_count, ax_count = plt.subplots(figsize=(8, 5))
        algorithms_list = list(community_counts.keys())
        count_values = list(community_counts.values())
        ax_count.bar(algorithms_list, count_values)
        ax_count.set_title("社团发现算法社团数量对比", fontproperties=font_prop)
        ax_count.set_xlabel("算法", fontproperties=font_prop)
        ax_count.set_ylabel("社团数量", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(fig_count, "community_count_comparison")

    def compare_algorithm_parameters(
        self,
        algorithm="louvain",
        parameter_name="resolution",
        parameter_values=[0.5, 1.0, 1.5],
    ):
        """比较特定算法不同参数下的结果"""
        print(f"\n--- {algorithm} 算法参数对比 ({parameter_name}) ---")
        modularity_scores = {}
        results = {}  # 存储不同参数下的社团发现结果
        community_counts = {}  # 存储不同参数下的社团数量
        avg_community_sizes = {}  # 存储不同参数下的平均社团规模
        avg_internal_densities = {}  # 存储不同参数下的平均内部密度
        avg_conductances = {}  # 存储不同参数下的平均电导率

        for value in parameter_values:
            print(f"运行 {algorithm} 算法，参数 {parameter_name}={value}...")
            partition = None
            modularity = None

            if algorithm == "louvain":
                # louvain 算法的参数通常是 resolution
                partition = community_louvain.best_partition(
                    self.graph, resolution=value
                )
                modularity = community_louvain.modularity(partition, self.graph)
                communities_dict = defaultdict(list)
                for node, comm_id in partition.items():
                    communities_dict[comm_id].append(node)
                communities_list = list(communities_dict.values())
            # 可以根据需要添加其他算法的参数调整

            if partition is not None:
                modularity_scores[value] = modularity
                results[value] = partition
                # 计算社团数量
                community_count = len(communities_list)
                num_communities = len(set(partition.values()))
                community_counts[value] = num_communities
                print(f"  发现社团数量: {num_communities}")
                if community_count > 0:
                    community_sizes_list = [len(comm) for comm in communities_list]
                    avg_community_size = np.mean(community_sizes_list)
                    avg_community_sizes[value] = avg_community_size
                    print(f"  平均社团规模: {avg_community_size:.4f}")
                    # 计算平均内部密度和平均电导率
                    internal_densities = []
                    conductances = []
                    for comm_nodes in communities_list:
                        if len(comm_nodes) > 1:
                            subgraph = self.graph.subgraph(comm_nodes)
                            internal_densities.append(nx.density(subgraph))
                            conductances.append(nx.conductance(self.graph, comm_nodes))
                    if internal_densities:
                        avg_internal_density = np.mean(internal_densities)
                        avg_internal_densities[value] = avg_internal_density
                        print(
                            f"  平均内部密度 (非单人社团): {avg_internal_density:.4f}"
                        )
                    else:
                        print("  没有非单人社团，无法计算平均内部密度。")
                    if conductances:
                        avg_conductance = np.mean(conductances)
                        avg_conductances[value] = avg_conductance
                        print(f"  平均电导率 (非单人社团): {avg_conductance:.4f}")
                    else:
                        print("  没有非单人社团，无法计算平均电导率。")

        print(f"\n{algorithm} 算法参数 ({parameter_name}) 模块度对比结果:")
        for value, mod in modularity_scores.items():
            print(f"  {parameter_name}={value}: {mod:.4f}")
        print(f"\n{algorithm} 算法参数 ({parameter_name}) 社团数量对比结果:")
        for value, count in community_counts.items():
            print(f"  {parameter_name}={value}: {count}")
        print(f"\n{algorithm} 算法参数 ({parameter_name}) 平均社团规模对比结果:")
        for value, size in avg_community_sizes.items():
            print(f"  {parameter_name}={value}: {size:.4f}")
        print(f"\n{algorithm} 算法参数 ({parameter_name}) 平均内部密度对比结果:")
        for value, density in avg_internal_densities.items():
            print(f"  {parameter_name}={value}: {density:.4f}")
        print(f"\n{algorithm} 算法参数 ({parameter_name}) 平均电导率对比结果:")
        for value, conductance in avg_conductances.items():
            print(f"  {parameter_name}={value}: {conductance:.4f}")

        # 绘制模块度对比图
        fig_modularity, ax_modularity = plt.subplots(figsize=(8, 5))
        param_values_str = [str(v) for v in parameter_values]
        modularity_values = [modularity_scores.get(v, 0) for v in parameter_values]
        ax_modularity.bar(param_values_str, modularity_values)
        ax_modularity.set_title(
            f"{algorithm} 算法参数 ({parameter_name}) 模块度对比",
            fontproperties=font_prop,
        )
        ax_modularity.set_xlabel(f"参数值 ({parameter_name})", fontproperties=font_prop)
        ax_modularity.set_ylabel("模块度", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(
            fig_modularity,
            f"{algorithm}_parameter_modularity_comparison_{parameter_name}",
        )
        # 绘制社团数量对比图
        fig_count, ax_count = plt.subplots(figsize=(8, 5))
        param_values_str = [str(v) for v in parameter_values]
        count_values = [community_counts.get(v, 0) for v in parameter_values]
        ax_count.bar(param_values_str, count_values)
        ax_count.set_title(
            f"{algorithm} 算法参数 ({parameter_name}) 社团数量对比",
            fontproperties=font_prop,
        )
        ax_count.set_xlabel(f"参数值 ({parameter_name})", fontproperties=font_prop)
        ax_count.set_ylabel("社团数量", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(
            fig_count,
            f"{algorithm}_parameter_community_count_comparison_{parameter_name}",
        )
        # 绘制平均社团规模对比图
        fig_size, ax_size = plt.subplots(figsize=(8, 5))
        param_values_str = [str(v) for v in parameter_values]
        size_values = [avg_community_sizes.get(v, 0) for v in parameter_values]
        ax_size.bar(param_values_str, size_values)
        ax_size.set_title(
            f"{algorithm} 算法参数 ({parameter_name}) 平均社团规模对比",
            fontproperties=font_prop,
        )
        ax_size.set_xlabel(f"参数值 ({parameter_name})", fontproperties=font_prop)
        ax_size.set_ylabel("平均社团规模", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(
            fig_size,
            f"{algorithm}_parameter_avg_community_size_comparison_{parameter_name}",
        )
        # 绘制平均内部密度对比图
        fig_density, ax_density = plt.subplots(figsize=(8, 5))
        param_values_str = [str(v) for v in parameter_values]
        density_values = [avg_internal_densities.get(v, 0) for v in parameter_values]
        ax_density.bar(param_values_str, density_values)
        ax_density.set_title(
            f"{algorithm} 算法参数 ({parameter_name}) 平均内部密度对比",
            fontproperties=font_prop,
        )
        ax_density.set_xlabel(f"参数值 ({parameter_name})", fontproperties=font_prop)
        ax_density.set_ylabel("平均内部密度", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(
            fig_density,
            f"{algorithm}_parameter_avg_internal_density_comparison_{parameter_name}",
        )
        # 绘制平均电导率对比图
        fig_conductance, ax_conductance = plt.subplots(figsize=(8, 5))
        param_values_str = [str(v) for v in parameter_values]
        conductance_values = [avg_conductances.get(v, 0) for v in parameter_values]
        ax_conductance.bar(param_values_str, conductance_values)
        ax_conductance.set_title(
            f"{algorithm} 算法参数 ({parameter_name}) 平均电导率对比",
            fontproperties=font_prop,
        )
        ax_conductance.set_xlabel(
            f"参数值 ({parameter_name})", fontproperties=font_prop
        )
        ax_conductance.set_ylabel("平均电导率", fontproperties=font_prop)
        plt.tight_layout()
        self._save_plot(
            fig_conductance,
            f"{algorithm}_parameter_avg_conductance_comparison_{parameter_name}",
        )
        # 对每个参数下的结果进行社团分析图表绘制
        for value, partition in results.items():
            print(
                f"\n--- {algorithm} 算法参数 {parameter_name}={value} 社团分析图表 ---"
            )
            # 社团规模分布
            community_sizes = defaultdict(int)
            for comm_id in partition.values():
                community_sizes[comm_id] += 1
            sorted_community_sizes = sorted(
                community_sizes.items(), key=lambda item: item[1], reverse=True
            )
            print("社团规模分布 (Top 10):", sorted_community_sizes[:10])
            self._plot_community_size_distribution(
                sorted_community_sizes,
                f"{algorithm} ({parameter_name}={value}) 社团规模分布 (Top 10)",
                f"{algorithm}_param_{parameter_name}_{value}_community_size_distribution_top10",
            )
            # 获取规模大于等于2的社团信息
            community_nodes_map = defaultdict(list)
            for node, comm_id in partition.items():
                community_nodes_map[comm_id].append(node)
            # 过滤出规模大于等于2的社团的 (ID, 规模) 元组列表
            communities_ge2 = [
                (community_id, len(nodes))
                for community_id, nodes in community_nodes_map.items()
                if len(nodes) >= 5
            ]
            communities_ge2_sorted = sorted(
                communities_ge2, key=lambda item: item[1], reverse=True
            )
            # 社团关系概览 (Top 10)
            self._plot_community_relationship_overview(
                partition,
                [comm_id for comm_id, size in communities_ge2_sorted],
                f"{algorithm} ({parameter_name}={value}) 社团关系概览",
                f"{algorithm}_param_{parameter_name}_{value}_community_relationship_overview_top10",
            )

            # 全部社团连接密度
            self._print_community_density(
                partition, f"{algorithm} ({parameter_name}={value}) 全部社团连接密度"
            )

            # 绘制其他社团分析图表
            self._plot_community_analysis(
                partition, f"{algorithm}_param_{parameter_name}_{value}"
            )

    def analyze_user_profiles_and_recommendations(
        self,
    ):
        """社交网络领域中用户画像与推荐系统分析"""
        print("\n--- 用户画像与推荐系统分析 ---")

        # 从节点角度 (保留原有分析)
        print("\n从节点角度:")
        # PageRank Top 10 作为较高重要性用户
        pagerank = nx.pagerank(self.graph)
        sorted_pagerank = sorted(
            pagerank.items(), key=lambda item: item[1], reverse=True
        )
        top_10_pagerank_users = [node for node, score in sorted_pagerank[:10]]
        print("PageRank Top 10 用户 (较高重要性用户):", top_10_pagerank_users)

        # 信息桥梁 (介数中心性 Top 10)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        sorted_betweenness = sorted(
            betweenness_centrality.items(), key=lambda item: item[1], reverse=True
        )
        top_10_bridges = [node for node, score in sorted_betweenness[:10]]
        print("信息桥梁 (介数中心性 Top 10):", top_10_bridges)

        # 影响力用户 (度中心性 Top 10)
        degree_centrality = nx.degree_centrality(self.graph)
        sorted_degree = sorted(
            degree_centrality.items(), key=lambda item: item[1], reverse=True
        )
        top_10_influencers = [node for node, score in sorted_degree[:10]]
        print("影响力用户 (度中心性 Top 10):", top_10_influencers)

        # 从社团角度 (使用 Louvain 算法，resolution=0.7)
        print("\n从社团角度 (Louvain 算法, resolution=0.7):")

        # 直接调用 Louvain 算法
        try:
            # networkx 0.99 及以上版本使用 community.louvain_communities
            # networkx 2.x 版本使用 community.best_partition
            # 这里假设使用较新版本
            import community as community_louvain

            partition = community_louvain.best_partition(self.graph, resolution=0.6)
            # 计算模块度 (可选，但通常会一起计算)
            modularity = community_louvain.modularity(partition, self.graph)
            print(f"Louvain 算法 (resolution=0.7) 模块度: {modularity:.4f}")

        except ImportError:
            print("请安装 python-louvain 库: pip install python-louvain")
            return
        except Exception as e:
            print(f"Louvain 算法执行出错: {e}")
            return

        if not partition:
            print("未能找到社团，跳过社团角度分析。")
            return

        # 构建社团ID到节点列表的映射
        community_nodes_map = defaultdict(list)
        for node, comm_id in partition.items():
            community_nodes_map[comm_id].append(node)

        # 按社团规模排序
        sorted_communities = sorted(
            community_nodes_map.items(), key=lambda item: len(item[1]), reverse=True
        )

        if not sorted_communities:
            print("没有发现社团。")
            return

        # 找到节点数量排名前 5 的社团
        num_top_communities = min(5, len(sorted_communities))
        top_5_communities = sorted_communities[:num_top_communities]
        # print(f"\n节点数量排名前 {num_top_communities} 的社团:")
        # for comm_id, nodes in top_5_communities:
        #     print(f"  社团 (ID: {comm_id}, 成员数: {len(nodes)})")

        # 在每个排名前 5 的社团中找意见领袖 (社团内部度中心性最高)
        print(f"\n排名前 {num_top_communities} 社团中的意见领袖:")
        top_influencers_in_top_communities = {}
        for comm_id, nodes in top_5_communities:
            if len(nodes) > 1:  # 确保社团成员数大于1才能计算度中心性
                subgraph = self.graph.subgraph(nodes)
                internal_degree_centrality = nx.degree_centrality(subgraph)
                if internal_degree_centrality:  # 确保计算结果不为空
                    sorted_internal_degree = sorted(
                        internal_degree_centrality.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    top_influencers_in_top_communities[
                        comm_id
                    ] = sorted_internal_degree[0][
                        0
                    ]  # 取度中心性最高的节点
                    print(
                        f"  社团 {comm_id} 意见领袖: {top_influencers_in_top_communities[comm_id]}"
                    )
                else:
                    print(
                        f"  社团 {comm_id} 成员数过少或无内部连接，无法确定意见领袖。"
                    )
            else:
                print(f"  社团 {comm_id} 成员数过少，无法确定意见领袖。")

        # 在最大社团中找重要用户 (社团内部 PageRank Top 10)
        largest_community_id, largest_community_nodes = sorted_communities[0]
        print(
            f"\n最大社团 (ID: {largest_community_id}) 中的重要用户 (PageRank Top 10):"
        )
        if len(largest_community_nodes) > 1:  # 确保社团成员数大于1才能计算 PageRank
            largest_subgraph = self.graph.subgraph(largest_community_nodes)
            internal_pagerank = nx.pagerank(largest_subgraph)
            if internal_pagerank:  # 确保计算结果不为空
                sorted_internal_pagerank = sorted(
                    internal_pagerank.items(), key=lambda item: item[1], reverse=True
                )
                top_10_internal_important_users = [
                    node for node, score in sorted_internal_pagerank[:10]
                ]
                print(top_10_internal_important_users)
            else:
                print("  最大社团成员数过少或无内部连接，无法确定重要用户。")
        else:
            print("  最大社团成员数过少，无法确定重要用户。")

        # 找边界用户 (与社团外部连接较多)
        # 边界用户可以通过计算每个节点连接到社团外部的边的数量来确定
        print("\n边界用户 (与社团外部连接较多 Top 10):")
        boundary_scores = {}
        for node in self.graph.nodes():
            comm_id = partition.get(node)  # 获取节点所属社团ID
            if comm_id is not None:  # 确保节点在某个社团中
                external_edges = 0
                for neighbor in self.graph.neighbors(node):
                    neighbor_comm_id = partition.get(neighbor)
                    if neighbor_comm_id is not None and neighbor_comm_id != comm_id:
                        external_edges += 1
                boundary_scores[node] = external_edges

        sorted_boundary_users = sorted(
            boundary_scores.items(), key=lambda item: item[1], reverse=True
        )
        top_10_boundary_users = [node for node, score in sorted_boundary_users[:10]]
        print(top_10_boundary_users)

        # 绘制图表可视化这三种用户
        self._plot_user_types(
            self.graph,
            top_influencers_in_top_communities,
            top_10_internal_important_users,
            top_10_boundary_users,
            partition,  # 传递社团划分信息用于着色
            "用户画像与推荐系统关键用户",
            "user_profiles_and_recommendations_key_users",
        )

    def _plot_user_types(
        self,
        graph,
        top_influencers_in_top_communities,
        top_10_internal_important_users,
        top_10_boundary_users,
        partition,
        title,
        filename,
    ):
        """
        绘制不同类型用户的图表。
        Args:
            graph (nx.Graph): 网络图。
            top_influencers_in_top_communities (dict): 顶级社团意见领袖字典 {社团ID: 节点ID}。
            top_10_internal_important_users (list): 最大社团重要用户列表。
            top_10_boundary_users (list): 边界用户列表。
            partition (dict): 社团划分字典 {节点ID: 社团ID}。
            title (str): 图表标题。
            filename (str): 保存文件名。
        """
        # print(f"绘制 {title}")

        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(graph)  # 使用整个网络的布局

        # 根据社团划分给节点着色
        node_colors = [
            partition.get(node, -1) for node in graph.nodes()
        ]  # -1 表示未分配社团的节点

        # 绘制所有节点 (根据社团着色)
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            cmap=plt.cm.viridis,  # 使用一个颜色映射
            node_size=50,
            alpha=0.6,
            label="普通用户",
        )

        # 突出显示不同类型的用户
        # 意见领袖 (排名前5社团各一个)
        influencer_nodes = list(top_influencers_in_top_communities.values())
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=influencer_nodes,
            node_color="red",
            node_size=200,
            label="社团意见领袖",
            edgecolors="black",  # 添加黑色边框
            linewidths=1.5,
        )

        # 重要用户 (最大社团 Top 10 PageRank)
        important_user_nodes = top_10_internal_important_users
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=important_user_nodes,
            node_color="gold",
            node_size=200,
            label="最大社团重要用户",
            edgecolors="black",  # 添加黑色边框
            linewidths=1.5,
        )

        # 边界用户 (与社团外部连接较多 Top 10)
        boundary_user_nodes = top_10_boundary_users
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=boundary_user_nodes,
            node_color="cyan",
            node_size=200,
            label="边界用户",
            edgecolors="black",  # 添加黑色边框
            linewidths=1.5,
        )

        # 绘制边 (可选，如果网络较大可能不绘制边)
        # nx.draw_networkx_edges(graph, pos, alpha=0.2)

        # 添加图例
        plt.legend(scatterpoints=1, loc="best", shadow=True, prop=font_prop)

        plt.title(title, fontproperties=font_prop)
        plt.axis("off")
        plt.tight_layout()
        self._save_plot(plt.gcf(), filename)  # 使用 plt.gcf() 获取当前图表对象
        plt.close()


# 主程序
if __name__ == "__main__":
    analyzer = SocialNetworkAnalyzer("assets/email-Eu-core.txt")

    # 第一部分：节点与边的重要性评估
    analyzer.analyze_node_importance()
    analyzer.analyze_edge_importance()

    # 第二部分：社团发现
    print("\n--- 开始社团发现 ---")
    louvain_partition, louvain_modularity = analyzer.find_communities(
        algorithm="louvain"
    )
    newman_partition, newman_modularity = analyzer.find_communities(algorithm="newman")
    label_propagation_partition, label_propagation_modularity = (
        analyzer.find_communities(algorithm="label_propagation")
    )

    # # 第三部分：比较与验证
    analyzer.compare_algorithms()

    # 选择一个算法进行参数对比，这里以 louvain 为例，调整 resolution 参数
    analyzer.compare_algorithm_parameters(
        algorithm="louvain",
        parameter_name="resolution",
        parameter_values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    )

    # 第四部分：社交网络领域中用户画像与推荐系统
    # 这里假设 louvain 是最好的算法，或者你可以根据模块度对比结果选择
    best_algorithm = "louvain"  # 根据模块度对比结果修改
    analyzer.analyze_user_profiles_and_recommendations()

    print("\n分析完成，图表已保存到 assets/pic 目录。")
