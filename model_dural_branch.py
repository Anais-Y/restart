import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score


class GAT(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_heads, dropout_disac, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout_disac)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout_disac)
        self.conv3 = GATConv(num_node_features, num_classes, heads=num_heads, concat=False, dropout=dropout_disac)
        self.bn1 = BatchNorm1d(num_node_features)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)

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
        x = global_mean_pool(x, batch)

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
        self.dataset = dataset
        if self.dataset == "DEAP":
            self.fusion_gat = nn.Linear(4 * num_classes, num_classes, bias=True)
        elif self.dataset == "SEED":
            self.fusion_gat = nn.Linear(5 * num_classes, 64, bias=True)
        else:
            raise ValueError("Please give a dataset")
        nn.init.kaiming_uniform_(self.fusion_gat.weight, nonlinearity='relu')
        self.de_encoder = SelfAttention(310*3, 64)  # 62*5*3
        self.fusion_all = nn.Linear(128, num_classes)
        self.ln = nn.LayerNorm(310*3)

    def forward(self, data):
        x_alpha = self.GAT_alpha(data['alpha'])
        x_beta = self.GAT_beta(data['beta'])
        x_gamma = self.GAT_theta(data['theta'])
        x_theta = self.GAT_gamma(data['gamma'])
        if self.dataset == "DEAP":
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        elif self.dataset == "SEED":
            x_delta = self.GAT_delta(data['delta'])
            x_concat = torch.cat((x_delta, x_alpha, x_beta, x_gamma, x_theta), dim=1)
        else:
            print('[Attention]!!!')
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        x_out = self.fusion_gat(x_concat)  # bs, 64
        x_out = F.elu(x_out)
        print(data['de'].shape)
        new_de_shape = (-1, data['de'].size(-3) * data['de'].size(-2) * data['de'].size(-1))
        de_feats = data['de'].view(*new_de_shape)
        de_feats = self.ln(de_feats)
        print(de_feats.shape)
        x_de = self.de_encoder(de_feats)  # bs, 64
        x_de = F.leaky_relu(x_de, negative_slope=1)
        # print(x_de.shape)
        x_out = self.fusion_all(torch.cat((x_out, x_de), dim=1))

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
