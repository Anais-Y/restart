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
        self.conv1 = GATv2Conv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout_disac)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout_disac)
        self.conv3 = GATv2Conv(num_node_features, hidden_dim, heads=num_heads, concat=False, dropout=dropout_disac)
        self.bn1 = BatchNorm1d(num_node_features)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)
        self.init_weights()
        self.hidden_dim = hidden_dim

    def init_weights(self):
        # 对GATConv层的权重进行初始化
        init.xavier_uniform_(self.conv3.att_src)
        init.xavier_uniform_(self.conv3.att_dst)
        if self.conv3.bias is not None:
            init.constant_(self.conv3.bias, 0)

    def forward(self, band_data):
        x, edge_index, batch = band_data.x, band_data.edge_index, band_data.batch

        # 第一层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn1(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)

        # 第二层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.bn2(x)
        # x = self.conv2(x, edge_index)
        # x = F.elu(x)
        # print(x.shape)
        # x = self.conv3(x, edge_index)
        # 全局平均池化
        # x = global_mean_pool(x, batch)
        bs = max(batch)+1
        x=x.reshape((bs, -1, self.hidden_dim))

        return x  # F.log_softmax(x, dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Initialize the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the divisors for the positional encoding
        div_term = torch.exp(torch.arange(0, d_model // 2).float() * -(np.log(10000.0) / d_model))
        
        # Extend div_term if necessary to match the dimension of d_model
        if d_model % 2 == 1:
            div_term = torch.cat([div_term, div_term[-1:]])
        
        # Apply sinusoidal functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:len(div_term) - (0 if d_model % 2 == 0 else 1)])
        
        # Register as a buffer to keep it fixed during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding
        # print("input shape:", x.size(), "pe size:", self.pe.size())
        x = x + self.pe[:x.size(1), :]
        return x
    

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
        self.dataset = dataset
        if self.dataset == "DEAP":
            self.fusion_gat = nn.Linear(4 * num_classes, num_classes, bias=True)
        elif self.dataset == "SEED":
            self.fusion_gat = nn.Linear(5 * num_classes, num_classes, bias=True)
            d_model = 5
        else:
            raise ValueError("Please give a dataset")
        nn.init.kaiming_uniform_(self.fusion_gat.weight, nonlinearity='relu')
        seq_length = 186
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_length)
        self.de_encoder = SelfAttention(in_features=186*15, hidden_dim=640)  # 62*5*3
        self.fusion_all = nn.Linear(128*5, num_classes)
        self.ln = nn.LayerNorm(d_model)
        self.pooling = nn.Linear(186, 1)

    def forward(self, data):
        x_alpha = self.GAT_alpha(data['alpha'])
        x_beta = self.GAT_beta(data['beta'])
        x_gamma = self.GAT_theta(data['theta'])
        x_theta = self.GAT_gamma(data['gamma'])
        # print(x_alpha.shape)  # bs, 3  
        if self.dataset == "DEAP":
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        elif self.dataset == "SEED":
            x_delta = self.GAT_delta(data['delta'])
            x_concat = torch.cat((x_delta, x_alpha, x_beta, x_gamma, x_theta), dim=2)  # bs*186, hiddim*5
            # print("x_concat", x_concat.shape)
        else:
            print('[Attention]!!!')
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        # x_out = self.fusion_gat(x_concat)  # bs, 64
        # x_out = F.elu(x_out)
        # de_shape = data['de'].shape
        # de_feats = data['de']
        # seq_length = de_shape[1]*de_shape[2]
        # new_de_shape = de_feats.reshape(de_shape[0], seq_length, de_shape[3])  # bs, 186, 5
        # encoded_data = self.pos_encoder(x_concat)
        # # de_feats = data['de'].view(*new_de_shape)
        # encoded_data = self.ln(encoded_data)
        bs, _, _ = x_concat.shape
        x_concat = x_concat.reshape((bs, -1))
        x_out = self.de_encoder(x_concat)
        # print("x_out", x_out.shape)
        x_out = self.fusion_all(x_out)
        # print(x_out.shape)
        # # print(encoded_data.shape)
        # x_de = self.de_encoder(encoded_data)  # bs, 186, 64
        # x_de = self.pooling(x_de.transpose(1, 2))  # bs, 64, 1
        # x_de = F.leaky_relu(x_de.reshape(-1, 64), negative_slope=1)
        # print(x_de.shape)
        # x_out = self.fusion_all(torch.cat((x_out, x_de), dim=1))

        return F.elu(x_out)


class MultiBandDataset(Dataset):
    def __init__(self, constructed):
        super(MultiBandDataset, self).__init__()
        self.constructed = constructed

    def __len__(self):
        return len(self.constructed['label'])

    def __getitem__(self, idx):
        band_list = list(self.constructed.keys())
        band_list.remove("label")
        sample = {band: self.constructed[band][idx] for band in band_list}
        label = self.constructed['label'][idx]
        de = self.constructed['de'][idx]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        if not isinstance(de, torch.Tensor):
            de = torch.tensor(de, dtype=torch.float32)
        sample['label'] = label
        sample['de'] = de
        return sample


def train(model, tr_loader, optimizer, criterion, device, max_grad):
    model.train()
    for training_data in tr_loader:
        # print(data)
        # print('train labels:', training_data['label'])
        labels = training_data['label'].to(device)
        training_data = {key: value.to(device) for key, value in training_data.items() if key != 'label'}
        optimizer.zero_grad()  # 清空梯度
        out = model(training_data)  # 前向传播
        # print(out.shape, labels.shape)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        optimizer.step()


def evaluate(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_loss = []
    with torch.no_grad():
        for testing_data in data_loader:
            # print('test labels:', testing_data['label'])
            labels = testing_data['label'].to(device)
            testing_data = {key: value.to(device) for key, value in testing_data.items() if key != 'label'}
            outputs = model(testing_data)
            loss = criterion(outputs, labels).item()
            # print(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_loss.append(loss)
            # print(loss)
    all_loss = sum(all_loss) / len(all_loss)
    # print(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1, all_loss



if __name__ == '__main__':
    def print_model_details(model, indent=0):
        for name, module in model.named_children():
            print('    ' * indent + f'{name}: {module}')
            if len(list(module.children())) > 0:
                print_model_details(module, indent + 1)


    model = FusionModel(num_node_features=200, hidden_dim=128, num_heads=4, dropout_disac=0.6, num_classes=3,
                        dataset='SEED')
    print_model_details(model)
