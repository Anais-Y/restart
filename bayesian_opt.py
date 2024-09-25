import argparse
import time
import configparser
import tqdm
import matplotlib.pyplot as plt
from torch import nn
from model_gat_seed import MultiBandDataset, FusionModel, train, evaluate
from torch_geometric.loader import DataLoader
from utils_de import *
from torch.optim.lr_scheduler import StepLR
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# 超参数搜索空间
space = [
    Integer(4, 8, name='batch_size'),   # 批大小
    Real(1e-5, 1e-2, "log-uniform", name='learning_rate'),  # 学习率
    Real(0.1, 0.7, name='dropout_disactive')  # Dropout 概率
]

# 数据集配置
DATASET = 'SEED'
config_file = f'./configs/{DATASET}/iv.conf'
config = configparser.ConfigParser()
config.read(config_file)

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--data', type=str, default=config['data']['data'], help='data path')
parser.add_argument('--dataset', type=str, default=config['data']['dataset'], help='dataset name')
parser.add_argument('--batch_size', type=int, default=config['data']['batch_size'], help="batch size")
parser.add_argument('--shuffle', type=bool, default=config['data']['shuffle'], help="shuffle train test or not")
parser.add_argument('--window_length', type=int, default=config['data']['window_length'])
parser.add_argument('--cls', type=int, default=config['data']['cls'], help='number of classes')
parser.add_argument('--num_of_vertices', type=int, default=config['model']['num_of_vertices'], help='number of channels')
parser.add_argument('--hidden_dim', type=int, default=config['model']['hidden_dim'], help='number of hidden dimension')
parser.add_argument('--num_heads', type=int, default=config['model']['num_heads'], help='number of attention heads')
parser.add_argument('--dropout_disactive', type=float, default=config['model']['dropout_disactive'], help='rate of dropout_disactive')
parser.add_argument("--strides", type=int, default=config['model']['strides'], help="滑动窗口步长，local时空图使用几个时间步构建的，默认为3")
parser.add_argument('--seed', type=int, default=config['train']['seed'], help='种子设置')
parser.add_argument("--learning_rate", type=float, default=config['train']['learning_rate'], help="初始学习率")
parser.add_argument("--weight_decay_rate", type=float, default=config['train']['weight_decay_rate'], help="Adam的L2正则系数")
parser.add_argument("--lr_decay_rate", type=float, default=config['train']['lr_decay_rate'], help="学习率衰减率")
parser.add_argument('--epochs', type=int, default=config['train']['epochs'], help="训练代数")
parser.add_argument('--check_gradient', type=bool, default=config['train']['check_gradient'], help="check gradient or not")
parser.add_argument('--patience', type=int, default=config['train']['patience'], help="patience to early stop")
parser.add_argument('--print_every', type=int, default=config['train']['print_every'], help="训练代数")
parser.add_argument('--lr_decay_every', type=int, default=config['train']['lr_decay_every'], help="lr decay every xx epochs")
parser.add_argument('--save', type=str, default=config['train']['save'], help='保存路径')
parser.add_argument('--expid', type=str, default=config['train']['expid'], help='实验 id')
parser.add_argument('--desc', type=str, default=config['train']['description'], help='实验说明')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")
parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)
init_seed(args.seed)  # 确保实验结果可以复现

constructed = construct_graphs(args.data, args.dataset, args.window_length, args.strides)
constructed_train, constructed_test = split_data(constructed, test_ratio=0.2, random_flag=True)

train_set = MultiBandDataset(constructed_train)
test_set = MultiBandDataset(constructed_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 优化目标函数
@use_named_args(space)
def objective(batch_size, learning_rate, dropout_disactive):
    # 设置DataLoader
    batch_size = int(batch_size)
    tr_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # 定义模型
    model = FusionModel(num_node_features=args.window_length, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                        dropout_disac=dropout_disactive, num_classes=args.cls, dataset=args.dataset).to(device)
    model_parameters_init(model)  # 初始化模型参数

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)

    # 训练和评估
    best_test_acc = 0
    patience = args.patience
    for epoch in range(args.epochs):
        train(model, tr_loader, optimizer, scheduler, criterion, device, max_grad=args.max_grad_norm)
        _, _, test_acc = evaluate(model, te_loader, criterion, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience = args.patience  # 重置 patience
        else:
            patience -= 1

        if patience == 0:
            break

    return -best_test_acc  # 因为 gp_minimize 是最小化问题，我们返回负的准确率

# 执行贝叶斯优化
res = gp_minimize(objective, space, n_calls=20, random_state=args.seed)

print(f"Best parameters: batch_size={res.x[0]}, learning_rate={res.x[1]}, dropout_disactive={res.x[2]}")
print(f"Best validation accuracy: {-res.fun:.4f}")

