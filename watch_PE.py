import argparse
import os
from model_gat_seed import MultiBandDataset, FusionModel, train, evaluate
from torch_geometric.data import DataLoader
from utils_de import *
import torch
import random
import wandb
import tqdm
import configparser
from torch.optim.lr_scheduler import StepLR

# config_file = 'configs/DEAP/s01.conf'
# start a new wandb run to track this script
def parse_conf_file(filepath):
    # 创建配置解析器
    config = configparser.ConfigParser()

    # 读取配置文件
    config.read(filepath)

    # 将配置项转换为字典
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    return config_dict


parser = argparse.ArgumentParser(description="Process a configuration file.")
parser.add_argument("--config_file", help="Path to the configuration file")
args, remaining_argv = parser.parse_known_args()
# Load configuration from file
if args.config_file:
    config = parse_conf_file(args.config_file)
else:
    config = {}
# parser = argparse.ArgumentParser(description='arguments')
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
args = parser.parse_args(remaining_argv)

wandb.init(
    # set the wandb project where this run will be logged
    project="0827-seed-deASgraph",
    # track hyperparameters and run metadata
    config=vars(args)
)

if not os.path.exists(args.save):
    os.makedirs(args.save)
init_seed(args.seed)  # 确保实验结果可以复现
constructed = construct_graphs(args.data, args.dataset, args.window_length, args.strides)
constructed_train, constructed_test = split_data(constructed, test_ratio=0.2, random_flag=False)
train_set = MultiBandDataset(constructed_train)
tr_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_set = MultiBandDataset(constructed_test)
te_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = FusionModel(num_node_features=args.window_length, hidden_dim=args.hidden_dim, num_heads=args.num_heads,
                    dropout_disac=args.dropout_disactive, num_classes=args.cls, dataset=args.dataset).to(device)
model_parameters_init(model)  # init parameters of model
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
wait = 0
te_acc_max = 0
for epoch in tqdm.tqdm(range(num_epochs)):
    if wait >= args.patience:
        log_string(log_file, f'early stop at epoch: {epoch:04d}')
        break
    train(model, tr_loader, optimizer, scheduler, criterion, device, args.max_grad_norm)
    train_acc, train_f1, tr_loss = evaluate(model, tr_loader, criterion, device)
    test_acc, test_f1, te_loss = evaluate(model, te_loader, criterion, device)
    infos = f'Epoch {epoch + 1}, Train Acc: {train_acc:.2f}, Train F1: {train_f1:.2f}, ' \
            f'Test Acc: {test_acc:.2f},Test F1: {test_f1:.2f}, '
    log_string(log_file, infos)
    print(infos)
    te_acc_max = max(te_acc_max, test_acc)

    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())})
            wandb.log({f"parameters/{name}": wandb.Histogram(param.data.cpu().numpy())})

    if te_loss <= te_loss_min:
        info1 = f'val loss decrease from {te_loss_min:.4f} to {te_loss:.4f}, ' \
                f'save model to ' \
                f'{args.save + "exp_" + str(args.expid) + "_" + str(round(te_loss, 2)) + "_best_model.pth"} '
        log_string(log_file, info1)
        print(info1)
        wait = 0  # 在这里把wait清零
        te_loss_min = te_loss
        state = {'state_dict': model.state_dict(), 'hyperparams': vars(args)}
        torch.save(state,
                   args.save + "exp_" + str(args.expid) + "_" + str(round(te_loss_min, 2)) + "_best_model.pth")
    else:
        wait += 1
    wandb.log({
        "epoch": epoch,
        "learning_rate": scheduler.get_last_lr()[0],
        "train_loss": tr_loss,
        "train_accuracy": train_acc,
        "val_loss": te_loss,
        "val_accuracy": test_acc,
    })
    # wandb.log({"roc_curve" : wandb.plot.roc_curve(te_labels, te_preds, labels=["LVLA", "HVLA", "LVHA" ,"HVHA"])})


    if args.check_gradient:
        check_grad(model, log_file)
    if (epoch+1) / print_every == 0:
        print(infos)

final_info =f'Test Acc: {te_acc_max}'
log_string(log_file, final_info)
print(final_info)
wandb.watch(model, criterion, log='all', log_freq=10)
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()



