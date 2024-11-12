import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool, TopKPooling
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class GAT(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_heads, dropout_disac, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout_disac)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout_disac)
        # self.conv3 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout_disac)
        self.bn1 = BatchNorm1d(num_node_features)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)
        self.pool1 = TopKPooling(hidden_dim * num_heads, ratio=0.5)
        self.pool2 = TopKPooling(num_classes, ratio=0.5)

    def forward(self, band_data):
        x, edge_index, batch = band_data.x, band_data.edge_index, band_data.batch

        # 第一层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn1(x)
        x_save, _tuple = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.relu(x_save)

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        # 第二层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        # x = self.conv3(x, edge_index)
        # 全局平均池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1), _tuple

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
        attn_weight_names = ['gamma', 'theta', 'beta', 'alpha', 'delta', 'de']
        for name in attn_weight_names:
            setattr(self, f'attn_weight_{name}', None)
        self.edge_index = None
        self.x_feature = None

        self.dataset = dataset
        if self.dataset == "DEAP":
            self.GAT_de = GAT(num_node_features=4, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
            self.fusion = nn.Linear(5 * num_classes, num_classes, bias=True)
        elif self.dataset == "SEED":
            self.GAT_de = GAT(num_node_features=5, hidden_dim=hidden_dim, num_heads=num_heads,
                             dropout_disac=dropout_disac, num_classes=num_classes)
            self.fusion = nn.Linear(6 * num_classes, num_classes, bias=True)
        else:
            raise ValueError("Please give a dataset")
        nn.init.kaiming_uniform_(self.fusion.weight, nonlinearity='relu')

    def forward(self, data):
        x_alpha, _tuple_alpha = self.GAT_alpha(data['alpha'])
        x_beta, _tuple_beta = self.GAT_beta(data['beta'])
        x_theta, _tuple_theta = self.GAT_theta(data['theta'])
        x_gamma, _tuple_gamma = self.GAT_gamma(data['gamma'])
        if self.dataset == "DEAP":
            x_de, _tuple_de = self.GAT_de(data['de'])
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta, x_de), dim=1)
            self.attn_weight_gamma = _tuple_gamma[1]
            self.attn_weight_theta = _tuple_theta[1]
            self.attn_weight_beta = _tuple_beta[1]
            self.attn_weight_alpha = _tuple_alpha[1]
            self.attn_weight_de = _tuple_de[1]
            self.edge_index = _tuple_gamma[0]
            self.x_feature = x_concat
        elif self.dataset == "SEED":
            x_delta, x_save = self.GAT_beta(data['beta'])
            x_de, _tuple_de = self.GAT_de(data['de'])
            self.attn_weight_gamma = _tuple_gamma[1]
            self.attn_weight_theta = _tuple_theta[1]
            self.attn_weight_beta = _tuple_beta[1]
            self.attn_weight_alpha = _tuple_alpha[1]
            # self.attn_weight_delta = _tuple_delta[1]
            self.attn_weight_de = _tuple_de[1]
            self.edge_index = _tuple_gamma[0]
            x_concat = torch.cat((x_delta, x_alpha, x_beta, x_gamma, x_theta, x_de), dim=1)
            self.x_feature = x_concat
        else:
            print('[Attention]!!!')
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        x_out = self.fusion(x_concat)

        return F.relu(x_out)


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
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        sample['label'] = label
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
    acc = {}
    with torch.no_grad():
        for i, testing_data in enumerate(data_loader):
            # print('test labels:', testing_data['label'])
            labels = testing_data['label'].to(device)
            testing_data = {key: value.to(device) for key, value in testing_data.items() if key != 'label'}
            outputs = model(testing_data)
            # print(outputs, model.attn_weight)
            torch.save(model.x_feature, f'/data/Anaiis/garage/vis_data/15_20131105/fusion_{i}.pt')
            # print(model.x_feature.shape)
            bands = ['gamma', 'theta', 'beta', 'alpha', 'de']
            for band in bands:
                # 使用 getattr 动态获取 model 的 attn_weight 属性
                attn_weight = getattr(model, f'attn_weight_{band}')
                
                # 动态生成文件路径
                torch.save(attn_weight, f'/data/Anaiis/garage/vis_data/15_20131105/attn_weight_l1{band}_{i}.pt')
            # # torch.save(model.attn_weight, f'/data/Anaiis/garage/vis_data/6_20130712/attn_weight_l1gamma_{i}.pt')
            torch.save(model.edge_index, f'/data/Anaiis/garage/vis_data/15_20131105/edge_index_l1_{i}.pt')
            torch.save(labels, f'/data/Anaiis/garage/vis_data/15_20131105/labels_{i}.pt')
            if labels != torch.load(f'/data/Anaiis/garage/vis_data/15_20131105/labels_{i}.pt').item():
                print("attn!")
            loss = criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        # np.save("/data/Anaiis/garage/vis_data/s25/labels0924.npy", all_labels)
        print("total samples:", i)
    
    # print(all_preds, all_labels)
    acc["all"] = np.sum(np.array(all_preds)==np.array(all_labels))/len(all_labels)
    predict_a = np.where(np.array(all_preds) <= 1, 0, 1)
    label_a = np.where(np.array(all_labels) <=1, 0, 1)
    acc["arousal"] = np.sum(predict_a==label_a)/len(label_a)
    predict_v = np.where(np.array(all_preds) % 2 == 0, 0, 1)
    label_v = np.where(np.array(all_labels) % 2 == 0, 0, 1)
    acc["valence"] = sum(predict_v==label_v)/len(label_a)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, f1, loss  # , acc, all_labels, all_preds


if __name__ == '__main__':
    def print_model_details(model, indent=0):
        for name, module in model.named_children():
            print('    ' * indent + f'{name}: {module}')
            if len(list(module.children())) > 0:
                print_model_details(module, indent + 1)


    model = FusionModel(num_node_features=12, hidden_dim=128, num_heads=16, dropout_disac=0.6, num_classes=4,
                        dataset='DEAP')
    print_model_details(model)
