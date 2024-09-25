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
from sklearn.model_selection import KFold
import numpy as np

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
parser.add_argument('--expid', type=str, default=config['train']['expid'], help='实验 id')
parser.add_argument('--desc', type=str, default=config['train']['description'], help='实验说明')
parser.add_argument('--max_grad_norm', type=float, default=config['train']['max_grad_norm'], help="梯度阈值")
parser.add_argument('--log_file', default=config['train']['log_file'], help='log file')
args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)
init_seed(args.seed)  # 确保实验结果可以复现

# 构建数据集
constructed = construct_graphs(args.data, args.dataset, args.window_length, args.strides)
dataset = MultiBandDataset(constructed)

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

fold = 0
all_test_accs = []
all_test_f1s = []

for train_index, test_index in kf.split(dataset):
    fold += 1
    print(f'Fold {fold}')

    # 根据索引划分训练集和测试集
    train_set = torch.utils.data.Subset(dataset, train_index)
    test_set = torch.utils.data.Subset(dataset, test_index)
    
    tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    te_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionModel(num_node_features=args.window_length, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                        dropout_disac=args.dropout_disactive, num_classes=args.cls, dataset=args.dataset).to(device)
    model_parameters_init(model)  # 初始化模型参数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay_rate)

    # 训练和评估
    num_epochs = args.epochs
    print_every = args.print_every
    te_loss_min = float('inf')
    max_te_acc = 0
    max_te_f1 = 0
    wait = 0
    
    for epoch in tqdm.tqdm(range(num_epochs)):
        if wait >= args.patience:
            print(f'Early stopping at epoch: {epoch:04d} for fold {fold}')
            break
        training_loss = train(model, tr_loader, optimizer, scheduler, criterion, device, max_grad=args.max_grad_norm)
        train_acc, train_f1, tr_loss = evaluate(model, tr_loader, criterion, device)
        test_acc, test_f1, te_loss = evaluate(model, te_loader, criterion, device)

        print(f'Epoch {epoch + 1}, Train Acc: {train_acc:.2f}, Train F1: {train_f1:.2f}, '
              f'Test Acc: {test_acc:.2f}, Test F1: {test_f1:.2f}')
        
        max_te_acc = max(max_te_acc, test_acc)
        max_te_f1 = max(max_te_f1, test_f1)

        # Early stopping机制
        if te_loss <= te_loss_min:
            te_loss_min = te_loss
            wait = 0
            state = {'state_dict': model.state_dict(), 'hyperparams': vars(args)}
            torch.save(state, f'{args.save}/fold_{fold}_best_model.pth')
        else:
            wait += 1

    print(f'Best test acc for fold {fold}: {max_te_acc:.2f}, Best F1: {max_te_f1:.2f}')
    all_test_accs.append(max_te_acc)
    all_test_f1s.append(max_te_f1)

# 输出交叉验证的平均结果
print(f'Average Test Accuracy: {np.mean(all_test_accs):.2f}')
print(f'Average Test F1 Score: {np.mean(all_test_f1s):.2f}')
