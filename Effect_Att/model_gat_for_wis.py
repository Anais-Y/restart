import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import argparse
import configparser
import tqdm
from utils import *
from torch.optim.lr_scheduler import StepLR



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
        x, attention_score1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)

        # 第二层GAT卷积
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.bn2(x)
        x, attention_score2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)

        # x = self.conv3(x, edge_index)
        # 全局平均池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1), attention_score1, attention_score2


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
        x_alpha, alpha_score1, alpha_score2 = self.GAT_alpha(data['alpha'])
        x_beta, beta_score1, beta_score2 = self.GAT_beta(data['beta'])
        x_gamma, gamma_score1, gamma_score2 = self.GAT_theta(data['theta'])
        x_theta, theta_score1, theta_score2 = self.GAT_gamma(data['gamma'])
        if self.dataset == "DEAP":
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        elif self.dataset == "SEED":
            x_delta = self.GAT_delta(data['delta'])
            x_concat = torch.cat((x_delta, x_alpha, x_beta, x_gamma, x_theta), dim=1)
        else:
            print('[Attention]!!!')
            x_concat = torch.cat((x_alpha, x_beta, x_gamma, x_theta), dim=1)
        x_out = self.fusion(x_concat)

        return F.relu(x_out), alpha_score1


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
    att_score = None
    for training_data in tr_loader:
        # print(data)
        # print('train labels:', training_data['label'])
        labels = training_data['label'].to(device)
        training_data = {key: value.to(device) for key, value in training_data.items() if key != 'label'}
        optimizer.zero_grad()  # 清空梯度
        out, att_score = model(training_data)  # 前向传播
        # print(out.shape, labels.shape)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        optimizer.step()
    return att_score



if __name__ == '__main__':
    # DATASET = 'DEAP'
    # config_file = f'./configs/{DATASET}/s01.conf'
    # config = configparser.ConfigParser()
    # config.read(config_file)
    # parser = argparse.ArgumentParser(description='arguments')
    # parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
    # parser.add_argument('--dataset', type=str, default=config['data']['dataset'], help='dataset name')
    # parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch size")
    # parser.add_argument('--shuffle', type=bool, default=config['data']['shuffle'], help="shuffle train test or not")
    # parser.add_argument('--window_length', type=int, default=config['data']['window_length'])
    # parser.add_argument('--cls', type=int, default=config['data']['cls'], help='number of classes')
    # parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'],
    #                     help='number of channels')
    # parser.add_argument('--hidden_dim', type=int, default=config['model']['hidden_dim'],
    #                     help='number of hidden dimension')
    # parser.add_argument('--num_heads', type=int, default=config['model']['num_heads'],
    #                     help='number of attention heads')
    # parser.add_argument('--dropout_disactive', type=float, default=config['model']['dropout_disactive'],
    #                     help='rate of dropout_disactive')
    # parser.add_argument("--strides", type=int, default=config['model']['strides'],
    #                     help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")
    # parser.add_argument('--seed', type=int, default=config['train']['seed'], help='种子设置')
    # parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'], help="初始学习率")
    # parser.add_argument("--weight_decay_rate", type=float, default=config['train']['weight_decay_rate'],
    #                     help="Adam的L2正则系数")
    # parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'], help="学习率衰减率")
    # parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="训练代数")
    # parser.add_argument('--check_gradient', type=bool, default=config['train']['check_gradient'],
    #                     help="check gradient or not")
    # parser.add_argument('--patience', type=int, default=config['train']['patience'], help="patience to early stop")
    # parser.add_argument('--print_every', type=int, default=config['train']['print_every'], help="训练代数")
    # parser.add_argument('--lr_decay_every', type=int, default=config['train']['lr_decay_every'],
    #                     help="lr decay every xx epochs")
    # parser.add_argument('--save', type=str, default=config['train']['save'], help='保存路径')
    # parser.add_argument('--expid', type=int, default=config['train']['expid'], help='实验 id')
    # parser.add_argument('--desc', type=str, default=config['train']['description'], help='实验说明')
    # parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")
    # parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')
    # args = parser.parse_args()
    #
    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    # init_seed(args.seed)  # 确保实验结果可以复现
    constructed = construct_graphs('./Data/len_96/False/s01/', 'DEAP', 96)
    constructed_train, constructed_test = split_data(constructed, test_ratio=0.2, random_flag=True)
    test_set = MultiBandDataset(constructed_test)
    te_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/micro/Anaiis/Effect_Att/garage/DEAP/s01exp_1_0.0_best_model.pth'
    model = FusionModel(num_node_features=96, hidden_dim=64, num_heads=8,
                        dropout_disac=0.6, num_classes=4, dataset='DEAP').to(device)
    saved_model = torch.load(model_path)
    model.load_state_dict(saved_model['state_dict'])
    model.eval()
    test_flag = True
    for testing_data in te_loader:
        if test_flag:
            test_flag = False
            labels = testing_data['label'].to(device)
            testing_data = {key: value.to(device) for key, value in testing_data.items() if key != 'label'}
            outputs, att_score = model(testing_data)
            dump_att_score = att_score[1].cpu().detach().numpy()
            print(dump_att_score.shape)



