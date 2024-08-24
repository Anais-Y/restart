import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.metrics import accuracy_score, f1_score
import numpy as np 


class GAT(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_heads, dropout_disac, num_classes):
        super(GAT, self).__init__()
        self.conv = GATv2Conv(num_node_features, hidden_dim, heads=1)
        self.conv2 = GATv2Conv(hidden_dim, 1, heads=1, concat=False, dropout=dropout_disac)
        self.hidden_dim = hidden_dim
        self.bn1 = BatchNorm1d(hidden_dim)

    def forward(self, band_data):
        x, edge_index, batch = band_data.x, band_data.edge_index, band_data.batch
        x = self.conv(x, edge_index)
        # x = x.mean(dim=-1)
        # print(x.shape) bs*62, 128
        # x = global_mean_pool(x, batch)
        x = self.bn1(x)
        x = self.conv2(x, edge_index)
        x = x.reshape((-1, 62))
        x = F.gelu(x)
        
        return x  # F.log_softmax(x, dim=1)
    

class SelfAttention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # 应用注意力得分
        return torch.matmul(attention_scores, V)
    

class FusionModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_heads, dropout_disac, num_classes, dataset):
        super(FusionModel, self).__init__()
        self.GAT_delta = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_alpha = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_beta = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                            dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_theta = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.GAT_gamma = GAT(num_node_features=num_node_features, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
        self.attn_delta = SelfAttention(in_features=62, hidden_dim=hidden_dim)  # 62*5*3
        self.attn_alpha = SelfAttention(in_features=62, hidden_dim=hidden_dim)  # 62*5*3
        self.attn_beta = SelfAttention(in_features=62, hidden_dim=hidden_dim)  # 62*5*3
        self.attn_theta = SelfAttention(in_features=62, hidden_dim=hidden_dim)  # 62*5*3
        self.attn_gamma = SelfAttention(in_features=62, hidden_dim=hidden_dim)  # 62*5*3
        self.dataset = dataset
        if self.dataset == "DEAP":
            self.fusion = nn.Linear(4 * hidden_dim, num_classes, bias=True)
        elif self.dataset == "SEED":
            self.fusion1 = nn.Linear(5 * hidden_dim, hidden_dim, bias=True)
            self.fusion2 = nn.Linear(hidden_dim, num_classes, bias=True)
        else:
            raise ValueError("Please give a dataset")
        nn.init.kaiming_uniform_(self.fusion1.weight, nonlinearity='relu')
        # self.ln = nn.LayerNorm(3*62)
        


    def forward(self, data):
        x_alpha = self.GAT_alpha(data['alpha'])
        # x_alpha = self.ln(x_alpha)
        x_alpha = self.attn_alpha(x_alpha)
        x_beta = self.GAT_beta(data['beta'])
        # x_beta = self.ln(x_beta)
        x_beta = self.attn_beta(x_beta)
        x_gamma = self.GAT_theta(data['theta'])
        # x_gamma = self.ln(x_gamma)
        x_gamma = self.attn_gamma(x_gamma)
        x_theta = self.GAT_gamma(data['gamma'])
        # x_theta = self.ln(x_theta)
        x_theta = self.attn_theta(x_theta)
        # print(x_alpha.shape) bs, hidden_dim
        if self.dataset == "DEAP":
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        elif self.dataset == "SEED":
            x_delta = self.GAT_delta(data['delta'])
            # x_delta = self.ln(x_delta)
            x_delta = self.attn_delta(x_delta)
            x_concat = torch.cat((x_delta, x_alpha, x_beta, x_gamma, x_theta), dim=1)
        else:
            print('[Attention]!!!')
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        # print("x_concat: ", x_concat.shape) bs, hidden_dim*5
        # x_concat = self.attn(x_concat)
        x_out = self.fusion1(x_concat)
        x_out = self.fusion2(x_out)

        return x_out