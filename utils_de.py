import os
import torch
import random
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split


def log_string(log, string):
    """打印log"""
    log.write(string + '\n')
    log.flush()


def count_parameters(model):
    """统计模型参数"""
    # for name, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(name)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_seed(seed):
    """Disable cudnn to maximize reproducibility 禁用cudnn以最大限度地提高再现性"""
    torch.cuda.cudnn_enabled = False
    """
    cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    然后再设置：torch.backends.cudnn.benchmark = True，当这个flag为True时，将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    但由于其是使用非确定性算法，这会让网络每次前馈结果略有差异,如果想要避免这种结果波动，可以将下面的flag设置为True
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def construct_graphs(dataset_dir, dataset, w_len, strides):
    """
    加载数据集，构建全连接的图，每三个样本构建一个图，存在字典里
    :param dataset_dir: 数据集目录
    :return constructed: 包含四个频带的数据和标签的字典
    """
    if dataset == "DEAP":
        channels = 32
        band_list = ['alpha', 'beta', 'gamma', 'theta']
        window_len = w_len
    elif dataset == "SEED":
        channels = 62
        band_list = ['delta', 'alpha', 'beta', 'gamma', 'theta']
        window_len = w_len
    else:
        raise ValueError("Please define a dataset")

    all_samples = np.load(os.path.join(dataset_dir + 'data.npy'))
    label = np.load(os.path.join(dataset_dir + 'label.npy'))
    de_feat = np.load(os.path.join(dataset_dir + 'de.npy'))
    # print(de_feat.shape)  # 3394, 62, 5 vs 3200, 32, 4
    constructed = {'label': [], 'de': []}
    edge_index_template = [[i, j] for i in range(strides * channels) for j in range(strides * channels) if i != j]
    edge_index_template = torch.tensor(edge_index_template, dtype=torch.long).t().contiguous()

    for i, band in enumerate(band_list):
        constructed[band] = []
        sample_band = np.squeeze(all_samples[:, :, i, :])  # (25600, 32, 12)
        for step in range(int(len(all_samples) / strides)):
            lab = np.unique(label[step * strides:(step + 1) * strides])
            de_sample = de_feat[step * strides:(step + 1) * strides]
            if len(lab) == 1:
                if band == 'alpha':
                    constructed['label'].append(lab[0])
                    # constructed['de'].append(de_sample)
                
                de_node_features = de_sample.reshape((-1, de_sample.shape[-1]))  # 将 de_sample 转为 2D
                de_node_features = torch.tensor(de_node_features, dtype=torch.float)
                de_data = Data(x=de_node_features, edge_index=edge_index_template.clone(),
                               y=torch.tensor(lab, dtype=torch.long))
                constructed['de'].append(de_data)
                
                node_features = sample_band[step * strides:(step + 1) * strides, :, :]
                node_features = node_features.reshape((-1, window_len))
                node_features = torch.tensor(node_features, dtype=torch.float)
                data = Data(x=node_features, edge_index=edge_index_template.clone(),
                            y=torch.tensor(lab, dtype=torch.long))
                constructed[band].append(data)
                # print(type(data))
                # print(constructed.keys())
        # print('constructed length:', len(constructed['label']))

    return constructed


def split_data(constructed_data, test_ratio, random_flag):
    # print(len(constructed_data['alpha']))
    bands = [list(f.values()) for f in
             [{k: v[i] for k, v in constructed_data.items() if k != 'label'}
              for i in range(len(constructed_data['label']))]]
    labels = constructed_data['label']
    print(random_flag)
    X_train, X_test, y_train, y_test = train_test_split(
        bands,
        labels,
        test_size=test_ratio,
        random_state=42,  # 随机种子，保证实验可重复
        shuffle=random_flag
    )
    # print('y_train shape:', len(y_train), y_train[0:200])
    bands_keys = list(constructed_data.keys())
    print(bands_keys)
    try:
        bands_keys.remove('label')
    except:
        raise ValueError("don't have <label> key")
    constructed_train = list_to_dict(X_train, y_train, bands_keys)
    constructed_test = list_to_dict(X_test, y_test, bands_keys)
    return constructed_train, constructed_test


def list_to_dict(features, labels, feature_keys):
    data_dict = {}
    print('list_dict len', len(features))
    for i, key in enumerate(feature_keys):
        data_dict[key] = [features[j][i] for j in range(len(features))]

    data_dict['label'] = labels
    return data_dict


def check_grad(model, log_f):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad_norm = parameter.grad.norm().item()
            log_string(log_f, f'{name} gradient: {grad_norm}')
            print(f'{name} gradient: {grad_norm}')
        else:
            log_string(log_f, f'{name} gradient: None')


def model_parameters_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p, gain=0.0003)
        else:
            nn.init.uniform_(p)


class StandardScaler:
    """标准转换器"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class NScaler:
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class MinMax01Scaler:
    """最大最小值01转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class MinMax11Scaler:
    """最大最小值11转换器"""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min


if __name__ == '__main__':
    constructed = construct_graphs('./Data/len_12/s01/')
    print(type(constructed), constructed.keys())
