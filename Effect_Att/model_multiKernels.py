import torch
import torch.nn as nn
import torch.nn.functional as F

class gcn_operation(nn.Module):
    def __init__(self, shared_FC, activation='leakyRelu'):
        """
        图卷积模块
        :param activation: 激活方式 {'relu', 'leakyRelu'}
        """
        super(gcn_operation, self).__init__()
        self.activation = activation
        self.FC = shared_FC

    def forward(self, x, constructed_adj):
        """
        :param x: (3*N, B, Cin)
        :param constructed_adj: (3*N, 3*N)
        :return: (3*N, B, Cout)
        """
        x = torch.einsum('nm, mbc->nbc', constructed_adj.to(x.device), x)
        if self.activation == 'relu':
            return torch.relu(self.FC(x))
        elif self.activation == 'leakyRelu':
            return F.leaky_relu(self.FC(x))


class STSGCM(nn.Module):
    def __init__(self, shared_FC, num_of_vertices, activation='leakyRelu'):
        """
        :param shared_FC: 共享权重
        :param num_of_vertices: 节点数量
        :param activation: 激活方式 {'relu', 'leakyRelu'}
        """
        super(STSGCM, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.activation = activation
        self.FC = shared_FC
        self.gcn_operation = gcn_operation(shared_FC, activation=activation)

    def forward(self, x, constructed_adj):
        """
        :param x: (3N, B, Cin)
        :param constructed_adj: (3N, 3N)
        :return: (N, B, Cout)
        """
        need_concat = []

        for i in range(3):  # 每个module里面有3个gcn
            x = self.gcn_operation(x, constructed_adj)
            need_concat.append(x)

        # shape of each element is (1, N, B, Cout)
        need_concat = [
            torch.unsqueeze(
                h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
            ) for h in need_concat
        ]  # crop操作

        out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
        # Crop之后再取最大值

        del need_concat

        return out


class STSGCL(nn.Module):
    def __init__(self,
                 history,
                 num_of_vertices,
                 in_dim,
                 strides=3,
                 activation='leakyRelu',
                 temporal_emb=True,
                 spatial_emb=True,
                 num_gcn=3):
        """
        :param history: 输入时间步长
        :param num_of_vertices: 节点数量
        :param in_dim: 输入维度
        :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param activation: 激活方式 {'relu', 'leakyRelu'}
        :param temporal_emb: 加入时间位置嵌入向量
        :param spatial_emb: 加入空间位置嵌入向量
        :param num_gcn: 并行的图卷积核数量
        """
        super(STSGCL, self).__init__()
        self.strides = strides
        self.history = history
        self.in_dim = in_dim
        self.num_of_vertices = num_of_vertices
        self.num_gcn = num_gcn
        self.constructed_adj = nn.ModuleList([BlockMatrix(num_of_vertices, strides) for _ in range(self.num_gcn)])
        self.shared_fc = nn.ModuleList([nn.Linear(self.in_dim, self.in_dim, bias=True) for _ in range(self.num_gcn)])
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.STSGCMS = nn.ModuleList([STSGCM(shared_FC=self.shared_fc[k], num_of_vertices=self.num_of_vertices,
                                             activation=self.activation) for k in range(self.num_gcn)])

        if self.temporal_emb:
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
            # 1, T, 1, Cin

        if self.spatial_emb:
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
            # 1, 1, N, Cin

        self.reset()  # 每一层初始化一个

    def reset(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x):
        """
        :param x: B, T, N, Cin
        :return: B, T-2, N, Cout
        """
        # print('num_gcn:', self.num_gcn)
        # print('fc', len(self.shared_fc))
        if self.temporal_emb:
            x = x + self.temporal_embedding

        if self.spatial_emb:
            x = x + self.spatial_embedding

        sum_out = []
        for k in range(self.num_gcn):
            adj = self.constructed_adj[k]()  # 调用BlockMatrix类初始化大邻接矩阵
            need_concat = []
            batch_size = x.shape[0]
            for i in range(self.history - self.strides + 1):
                t = x[:, i: i+self.strides, :, :]  # (B, 3, N, Cin) 滑动
                t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
                # (B, 3*N, Cin)
                t = self.STSGCMS[k](t.permute(1, 0, 2), adj)
                # (3*N, B, Cin) -> (N, B, Cout) 聚合信息成一个片
                t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)
                # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
                need_concat.append(t)

            out = torch.cat(need_concat, dim=1)  # (B, T-2, N, Cout)
            sum_out.append(out)

            del need_concat, batch_size, out

        return torch.sum(torch.stack(sum_out), dim=0)  # 元素相加，把并行的三个输出加起来


class BlockMatrix(nn.Module):
    def __init__(self, size, strides):
        super(BlockMatrix, self).__init__()
        self.N = size
        self.strides = strides
        self.A = nn.Parameter(torch.FloatTensor(size, size))
        self.weight_cross_time = nn.Parameter(torch.FloatTensor(strides - 1, size, size))
        self.weight_cross_time2 = nn.Parameter(torch.FloatTensor(size, size))

    def forward(self):
        N = self.N
        local_adj = torch.zeros((N * self.strides, N * self.strides))
        for i in range(self.strides):
            local_adj[i * N:(i + 1) * N, i * N: (i + 1) * N] = self.A
        for k in range(self.strides - 1):
            local_adj[k*N:(k+1)*N, (k+1)*N:(k+2)*N] = self.weight_cross_time[k]  # 可更新的权重
            local_adj[(k+1)*N:(k+2)*N, k*N:(k+1)*N] = self.weight_cross_time[k].T
        for j in range(self.strides-2):
            local_adj[j * N:(j+1) * N, (j+2) * N: (j+3) * N] = self.weight_cross_time2
            local_adj[(j+2) * N: (j+3) * N, j * N:(j+1) * N] = self.weight_cross_time2.T
        return local_adj


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


class output_layer(nn.Module):
    def __init__(self, num_of_vertices, history, in_dim,
                 hidden_dim=128):
        """
        预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
        :param num_of_vertices:节点数
        :param history:输入时间步长
        :param in_dim: 输入维度
        :param hidden_dim:中间层维度
        """
        super(output_layer, self).__init__()
        self.num_of_vertices = num_of_vertices  # 14
        self.history = history  # 3
        self.in_dim = in_dim  # 128
        self.hidden_dim = hidden_dim  # 128 by default

        self.FC1 = nn.Linear(self.num_of_vertices * self.hidden_dim, self.hidden_dim, bias=True)
        self.self_attention = SelfAttention(in_dim * history, hidden_dim)

        self.FC2 = nn.Linear(self.hidden_dim, 2, bias=True)

    def forward(self, x):
        """
        :param x: (B, Tin, N, Cin)
        :return: (B)
        """
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin
        x = x.reshape(batch_size, self.num_of_vertices, self.history*self.in_dim)  # (B*N, Tin, Cin)
        x = self.self_attention(x)  # Apply self-attention
        # out1 = x.reshape(batch_size, self.num_of_vertices * self.hidden_dim)  # Reshape to (B, N*hidden_dim)
        out1 = F.leaky_relu(self.FC1(x.reshape(batch_size, -1)))
        out2 = self.FC2(out1)  # (B, N*hidden_dim) -> (B, 2)

        del out1, batch_size

        return torch.squeeze(out2)  # B


class STSGCN(nn.Module):
    def __init__(self, history, num_of_vertices, in_dim, num_layers,
                 first_layer_embedding_size, out_layer_dim, activation='leakyRelu',
                 temporal_emb=True, spatial_emb=True, strides=3, num_gcn=3):
        """
        :param history:输入时间步长
        :param num_of_vertices:节点数量
        :param in_dim:输入维度
        :param num_layers:几层STSGCL
        :param first_layer_embedding_size: 第一层输入层的维度
        :param out_layer_dim: 输出模块中间层维度
        :param activation: 激活函数 {relu, leakyRelu}
        :param temporal_emb:是否使用时间嵌入向量
        :param spatial_emb:是否使用空间嵌入向量
        :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为3
        :param num_gcn: 并行的图卷积核数量
        """
        super(STSGCN, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.num_layers = num_layers
        self.out_layer_dim = out_layer_dim
        self.activation = activation
        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb
        self.strides = strides
        self.num_gcn = num_gcn
        self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
        self.bn1 = nn.BatchNorm2d(num_features=9)
        self.STSGCLS = nn.ModuleList()
        self.STSGCLS.append(
            STSGCL(
                history=history,
                num_of_vertices=self.num_of_vertices,
                in_dim=first_layer_embedding_size,
                strides=self.strides,
                activation=self.activation,
                temporal_emb=self.temporal_emb,
                spatial_emb=self.spatial_emb,
                num_gcn=self.num_gcn
            )
        )
        history -= (self.strides - 1)  # 每经过一次GCL特征T都会少2
        for idx in range(1, self.num_layers):
            self.STSGCLS.append(
                STSGCL(
                    history=history,
                    num_of_vertices=self.num_of_vertices,
                    in_dim=first_layer_embedding_size,
                    strides=self.strides,
                    activation=self.activation,
                    temporal_emb=self.temporal_emb,
                    spatial_emb=self.spatial_emb,
                    num_gcn=self.num_gcn
                )
            )
            history -= (self.strides - 1)

        self.predictLayer = output_layer(
                    num_of_vertices=self.num_of_vertices,
                    history=history,
                    in_dim=first_layer_embedding_size,
                    hidden_dim=out_layer_dim
                )

    def forward(self, x):
        """
        :param x: B, Tin, N, Cin)
        :return: B
        """

        x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin
        x = self.bn1(x)
        for model in self.STSGCLS:
            x = model(x)
        # (B, T - 8, N, Cout)  因为有4个串联的STSGCL
        out = self.predictLayer(x)  # (B, )

        return out
