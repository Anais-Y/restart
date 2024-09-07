import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

bands = ['gamma', 'theta', 'beta', 'alpha', 'de']
person = "15_20131105"
dataset = "seed"
for band in bands:
    for n in range(225):
    # 加载注意力权重和 edge_index
        attention_weights = torch.load(f'/data/Anaiis/garage/vis_data/{person}/attn_weight_l1{band}_{n}.pt')
        edge_index = torch.load(f'/data/Anaiis/garage/vis_data/{person}/edge_index_l1_{n}.pt')
        label = torch.load(f'/data/Anaiis/garage/vis_data/{person}/labels_{n}.pt').item()

        # 将 edge_index 和 attention_weights 转换为 NumPy 数组
        edge_index_np = edge_index.cpu().numpy()
        attention_weights_np = attention_weights.cpu().numpy().squeeze()

        # 计算每个边的平均注意力权重（如果有多个头部）
        attention_weights_np = attention_weights_np.mean(axis=1)

        # 创建一个无向图
        G = nx.Graph()

        # 假设节点编号从0开始连续编号
        num_nodes = edge_index.max().item() + 1
        G.add_nodes_from(range(num_nodes))

        # 添加边及其权重
        for i in range(edge_index_np.shape[1]):
            src = edge_index_np[0, i]
            dst = edge_index_np[1, i]
            weight = attention_weights_np[i]
            G.add_edge(src, dst, weight=weight)

        # 计算每个节点的总连接强度（即所有相连边的权重之和）
        node_strength = {node: 0.0 for node in G.nodes()}
        for (u, v, d) in G.edges(data=True):
            node_strength[u] += d['weight']
            node_strength[v] += d['weight']

        # 固定节点位置
        pos = nx.spring_layout(G, seed=42)  # 固定布局

        # 将节点强度转换为颜色值（归一化到 [0, 1]）
        node_colors = np.array(list(node_strength.values()))
        node_colors_normalized = (node_colors - node_colors.min()) / (node_colors.max() - node_colors.min())
        cmap = plt.cm.viridis

        # 绘制图形
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_normalized, node_size=500, cmap=cmap)
        weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, width=1, edge_color='grey', alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=node_colors.min(), vmax=node_colors.max()))
        sm.set_array([])
        plt.colorbar(sm, label='Node Connection Strength')

        plt.title('EEG Graph Visualization with Attention Weights')
        # plt.show()
        filepath = f'/data/Anaiis/garage/figs/{dataset}_label{label}/{person}/'
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        plt.savefig(os.path.join(filepath,f'{band}_fig_{n}'))
    
