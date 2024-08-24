import torch.optim as optim
from model_multiKernels import *
import utils


class trainer():
    def __init__(self, args, scaler, history, num_of_vertices,
                 in_dim, num_layers, first_layer_embedding_size, out_layer_dim,
                 log, lrate, device, activation='leakyRelu', max_grad_norm=5,
                 lr_decay=False, temporal_emb=True, spatial_emb=True, strides=3, num_gcn=3):
        """
        训练器
        :param args: 参数脚本
        :param scaler: 转换器
        :param history: 输入时间步长
        :param num_of_vertices: 节点数量
        :param in_dim: 输入维度
        :param num_layers:几层STSGCL
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param log: 日志
        :param lrate: 初始学习率
        :param device: 计算设备
        :param activation:激活函数 {relu, GlU}
        :param max_grad_norm: 梯度阈值
        :param lr_decay: 是否采用初始学习率递减策略
        :param temporal_emb: 是否使用时间嵌入向量
        :param spatial_emb: 是否使用空间嵌入向量
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_gcn: 并行的图卷积核数量
        """
        super(trainer, self).__init__()

        self.model = STSGCN(
            history=history,
            num_of_vertices=num_of_vertices,
            in_dim=in_dim,
            num_layers=num_layers,
            first_layer_embedding_size=first_layer_embedding_size,
            out_layer_dim=out_layer_dim,
            activation=activation,
            temporal_emb=temporal_emb,
            spatial_emb=spatial_emb,
            strides=strides,
            num_gcn=num_gcn
        )

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)

        self.model.to(device)

        self.model_parameters_init()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, eps=1.0e-8, weight_decay=0, amsgrad=False)

        torch.autograd.set_detect_anomaly(True)

        if lr_decay:
            utils.log_string(log, 'Applying learning rate decay.')
            lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                     milestones=lr_decay_steps,
                                                                     gamma=args.lr_decay_rate)
        self.loss = torch.nn.SmoothL1Loss()
        self.l1_loss = torch.nn.L1Loss(size_average=False)
        self.lambda_l1 = 0
        self.scaler = scaler
        self.clip = max_grad_norm

        utils.log_string(log, "模型可训练参数: {:,}".format(utils.count_parameters(self.model)))
        utils.log_string(log, 'GPU使用情况:{:,}'.format(
            torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0))


    def compute_loss(self, preds, labels, adj_weights):
        """
        计算包含L1正则化的损失函数
        :param preds: 模型的输出
        :param labels: 真实标签
        :param adj_weights: 邻接矩阵的权重
        """
        loss = self.loss(preds, labels)  # 原始损失函数
        l1_reg = 0  # 初始化 L1 正则化为 0
        for adj_weight in adj_weights:  # 对每个邻接矩阵权重计算 L1 正则化
            l1_reg += self.lambda_l1 * self.l1_loss(adj_weight, torch.zeros_like(adj_weight))
        total_loss = loss + l1_reg  # 总损失
        return total_loss

    def model_parameters_init(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.0003)
            else:
                nn.init.uniform_(p)

    def train(self, input, real_val):
        """

        """

        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input)  # B
        # predict = self.scaler.inverse_transform(predict)

        # 如果scaler选择了None, inverse transform和transform都对我没作用
        # adj_weights = []
        # for name, param in self.model.named_parameters():
        #     if 'adj' in name or 'weight_cross_time' in name:
        #         adj_weights.append(param)
        #  break

        loss = self.loss(predict, real_val)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            # 防止梯度爆炸

        self.optimizer.step()

        mae = utils.masked_mae(predict, real_val).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()
        acc_all = utils.accuracy(predict, real_val)['all']
        acc_val = utils.accuracy(predict, real_val)['valence']
        acc_arou = utils.accuracy(predict, real_val)['arousal']
        return loss.item(), mae, mape, rmse, acc_all, acc_val, acc_arou

    def evel(self, input, real_val):
        """
        x shape:  (16969, 12, 307, 1) , y shape:  (16969, 12, 307)
        :param input: B, T, N, C
        :param real_val:B
        """
        self.model.eval()

        predict = self.model(input)  # B
        # predict = self.scaler.inverse_transform(predict)
        # print(predict, '\n real:', real_val)
        mae = utils.masked_mae(predict, real_val, 0.0).item()
        mape = utils.masked_mape(predict, real_val, 0.0).item()
        rmse = utils.masked_rmse(predict, real_val, 0.0).item()
        acc_all = utils.accuracy(predict, real_val)['all']
        acc_val = utils.accuracy(predict, real_val)['valence']
        acc_arou = utils.accuracy(predict, real_val)['arousal']
        return mae, mape, rmse, acc_all, acc_val, acc_arou
