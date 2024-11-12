import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from utils_de import *
from thop import profile


class GAT(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_heads, dropout_disac, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout_disac)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout_disac)
        # self.conv3 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout_disac)
        self.bn1 = BatchNorm1d(num_node_features)
        self.bn2 = BatchNorm1d(hidden_dim * num_heads)

    def forward(self, band_data):
        x, edge_index, batch = band_data.x, band_data.edge_index, band_data.batch

        # 第一层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 第二层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)

        # x = self.conv3(x, edge_index)
        # 全局平均池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


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
            self.fusion = nn.Linear(4 * num_classes, num_classes, bias=True)
        elif self.dataset == "SEED":
            self.fusion = nn.Linear(5 * num_classes, num_classes, bias=True)
        else:
            raise ValueError("Please give a dataset")
        nn.init.kaiming_uniform_(self.fusion.weight, nonlinearity='relu')

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


def train(model, tr_loader, optimizer, scheduler, criterion, device, max_grad):
    model.train()
    for training_data in tr_loader:
        # print(data)
        # print('train labels:', training_data['label'])
        labels = training_data['label'].to(device)
        training_data = {key: value.to(device) for key, value in training_data.items() if key != 'label'}
        optimizer.zero_grad()  # 清空梯度
        out = model(training_data)  # 前向传播
        flops, params = profile(model, inputs=(training_data, ))
        print(flops, params)
        # print(out.shape, labels.shape)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        optimizer.step()
    scheduler.step()


def evaluate(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    acc = {}
    with torch.no_grad():
        for testing_data in data_loader:
            # print('test labels:', testing_data['label'])
            labels = testing_data['label'].to(device)
            testing_data = {key: value.to(device) for key, value in testing_data.items() if key != 'label'}
            outputs = model(testing_data)
            loss = criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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


    model = FusionModel(num_node_features=96, hidden_dim=128, num_heads=4, dropout_disac=0.6, num_classes=4,
                        dataset='DEAP')
    print_model_details(model)
    print("模型可训练参数: {:,}".format(count_parameters(model)))
