import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


file_path = './channel-order.xlsx'
sheet1_df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
channel_index_dict = {idx: channel_name for idx, channel_name in enumerate(sheet1_df.iloc[:, 0])}
channel_name_list = list(channel_index_dict.values())

channels_to_remove = ['PO5', 'PO6', 'CB1', 'CB2']
indices_to_remove = []
for chan in channels_to_remove:
    index = channel_name_list.index(chan)
    indices_to_remove.append(index)
channel_name_list = np.delete(channel_name_list, indices_to_remove, axis=0)

bands = ["alpha", "beta", "theta", "gamma", "de"]
# selec_lab = 2
person = '6_20130712'
for selec_lab in [0, 1, 2]:
    for band in bands:
        # band = "beta"
        cnt = 0
        edge_index_np = None
        attention_weights_np = None 
        for n in range(225):
            attention_weights = torch.load(f'/data/Anaiis/garage/vis_data/{person}/attn_weight_l2{band}_{n}.pt')
            edge_index = torch.load(f'/data/Anaiis/garage/vis_data/{person}/edge_index_l2_{n}.pt')
            label = torch.load(f'/data/Anaiis/garage/vis_data/{person}/labels_{n}.pt').item()
            if label == selec_lab :
                # 将 edge_index 和 attention_weights 转换为 NumPy 数组
                if cnt == 0:
                    edge_index_np = edge_index.cpu().numpy()
                    attention_weights_np = attention_weights.cpu().numpy().squeeze()
                    cnt += 1
                else:
                    attention_weights_np += attention_weights.cpu().numpy().squeeze()
        num_nodes = edge_index_np.max() + 1  # 节点个数
        adj_matrix = np.zeros((num_nodes, num_nodes))

        for idx in range(edge_index_np.shape[1]):
            i, j = edge_index_np[:, idx]
            adj_matrix[i, j] = attention_weights_np[idx].mean()

        degrees = adj_matrix.mean(axis = 1)

        degrees = degrees.reshape((-1, 62))
        print(degrees.shape)
        degrees_selected_chans = np.zeros((3, 58))
        for t in range(3):
            degrees_selected_chans[t] = np.delete(degrees[t], indices_to_remove, axis=0)

        print("shape after remove: ", degrees_selected_chans.shape)
        scaler = MinMaxScaler()
        degrees_minmax = scaler.fit_transform(degrees_selected_chans)

        # Log Normalization (加1以避免 log(0))
        degrees_log = np.log(degrees_selected_chans + 1)

        # Standard (Z-score) Normalization
        degrees_std = (degrees_selected_chans - np.mean(degrees_selected_chans, axis=0)) / np.std(degrees_selected_chans, axis=0)


        montage = mne.channels.make_standard_montage('biosemi64')
        channels = list(channel_name_list)
        info = mne.create_info(ch_names=channels, sfreq=1, ch_types='eeg')
        info.set_montage(montage)

        data = np.zeros((58, 1))
        raw = mne.io.RawArray(data, info)

        norm = Normalize(vmin=np.min(degrees_selected_chans), vmax=np.max(degrees_selected_chans))

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 定义三个时间点的 degree_array 和各自的 axes
        degree_arrays = [degrees_selected_chans[0], degrees_selected_chans[1], degrees_selected_chans[2]]
        titles = ['T0', 'T1', 'T2']

        # 循环绘制每个图
        for i, (degree_array, ax) in enumerate(zip(degree_arrays, axes)):
            # 使用统一的颜色映射
            raw.plot_sensors(ch_type="eeg", axes=ax, show=False)
            sensor_points = ax.collections[0]
            positions = sensor_points.get_offsets()

            # 设置颜色
            colors = [plt.cm.inferno(norm(degree_array[i])) for i in range(58)]
            sensor_points.set_facecolors(colors)
            sensor_points.set_edgecolors('k')
            sensor_points.set_linewidths(0.8)

            # 设置节点大小（加大点大小）
            sensor_points.set_sizes([200] * len(positions))  # 200 是调整后的大小，可以根据需要增减

            # 显示通道名称
            for j, ch_name in enumerate(channels):
                ax.text(positions[j, 0], positions[j, 1] + 0.008, ch_name, fontsize=8, ha='center', va='center')

            # 添加标题
            ax.set_title(titles[i])

        # 在右侧添加一个共享的颜色条
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
        sm.set_array([])

        # 调整子图布局，确保colorbar不会与图片重叠
        fig.subplots_adjust(right=0.85)  # 为colorbar预留空间，右侧留出15%的宽度

        # 将 colorbar 放在右侧
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(sm, cax=cbar_ax, label="density")

        # 调整布局并显示图像
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 保证子图不被重叠
        plt.show()
        plt.savefig(f'electron/fig-label{selec_lab}-{person}-{band}-l2')