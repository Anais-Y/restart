import argparse
import time
import configparser
import tqdm
import matplotlib.pyplot as plt
from torch import nn
from model_gat_vis import MultiBandDataset, FusionModel, train, evaluate
from torch_geometric.data import DataLoader
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils_de import *
from torch.optim.lr_scheduler import StepLR

DATASET = 'SEED'
config_file = f'./s01.conf'
config = configparser.ConfigParser()
config.read(config_file)
parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
parser.add_argument('--dataset', type=str, default=config['data']['dataset'], help='dataset name')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch size")
parser.add_argument('--shuffle', type=bool, default=config['data']['shuffle'], help="shuffle train test or not")
parser.add_argument('--window_length', type=int, default=config['data']['window_length'])
parser.add_argument('--cls', type=int, default=config['data']['cls'], help='number of classes')
parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'],
                    help='number of channels')
parser.add_argument('--hidden_dim', type=int, default=config['model']['hidden_dim'],
                    help='number of hidden dimension')
parser.add_argument('--num_heads', type=int, default=config['model']['num_heads'],
                    help='number of attention heads')
parser.add_argument('--dropout_disactive', type=float, default=config['model']['dropout_disactive'],
                    help='rate of dropout_disactive')
parser.add_argument("--strides", type=int, default=config['model']['strides'],
                    help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")
parser.add_argument('--seed', type=int, default=config['train']['seed'], help='种子设置')
parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'], help="初始学习率")
parser.add_argument("--weight_decay_rate", type=float, default=config['train']['weight_decay_rate'], help="Adam的L2正则系数")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'], help="学习率衰减率")
parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="训练代数")
parser.add_argument('--check_gradient', type=bool, default=config['train']['check_gradient'],
                    help="check gradient or not")
parser.add_argument('--patience', type=int, default=config['train']['patience'], help="patience to early stop")
parser.add_argument('--print_every', type=int, default=config['train']['print_every'], help="训练代数")
parser.add_argument('--lr_decay_every', type=int, default=config['train']['lr_decay_every'], help="lr decay every xx epochs")
parser.add_argument('--save', type=str, default=config['train']['save'], help='保存路径')
parser.add_argument('--expid', type=int, default=config['train']['expid'], help='实验 id')
parser.add_argument('--desc', type=str, default=config['train']['description'], help='实验说明')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")
parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)
init_seed(args.seed)  # 确保实验结果可以复现
constructed = construct_graphs(args.data, args.dataset, args.window_length, args.strides)
test_set = MultiBandDataset(constructed)
print(len(test_set))
te_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = FusionModel(num_node_features=args.window_length, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                    dropout_disac=args.dropout_disactive, num_classes=args.cls, dataset=args.dataset).to(device)
ckpt = torch.load('/data/Anaiis/garage/DEAPnoshufTopK/exp_6-1004_0.0_best_model.pth')
model.load_state_dict(ckpt['state_dict'], strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay_rate)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)

num_epochs = args.epochs
print_every = args.print_every
log_file = open(args.log_file+args.desc, 'w')
log_string(log_file, str(args))
log_string(log_file, "模型可训练参数: {:,}".format(count_parameters(model)))
log_string(log_file, 'GPU使用情况:{:,}'.format(
    torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0))

te_loss_min = float('inf')

test_acc, test_f1, te_loss = evaluate(model, te_loader, criterion, device)
infos = f'Test Acc: {test_acc:.2f}, Test F1: {test_f1:.2f}'
log_string(log_file, infos)
print(infos)

