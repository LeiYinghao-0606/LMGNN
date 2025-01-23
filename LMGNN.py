import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from Params import args  # 确保 args 包含所有必要的参数
from mamba_ssm import Mamba
from Utils.loss_torch import bpr_loss, l2_reg_loss, InfoNCE  # 确保这些函数可用

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.35):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super(MambaLayer, self).__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(1)  # (batch_size, 1, hidden_size)
        hidden_states = self.mamba(input_tensor)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.ffn(hidden_states)
        if hidden_states.shape[1] == 1:
            hidden_states = hidden_states.squeeze(1)  # (batch_size, hidden_size)
        return hidden_states

class GNNLayer(nn.Module):
    def __init__(self):
        super(GNNLayer, self).__init__()

    def forward(self, adj, embeds):
        # 确保邻接矩阵和嵌入在同一设备上
        adj = adj.to(embeds.device)
        return torch.sparse.mm(adj, embeds)

class MambaTransGNN_SelfSupervised(nn.Module):
    def __init__(self):
        super(MambaTransGNN_SelfSupervised, self).__init__()
        #self.scale = nn.Parameter(torch.ones(args.latdim))
        #self.shift = nn.Parameter(torch.zeros(args.latdim))
        # 初始化用户和物品嵌入
        self.user_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.user, args.latdim)))
        self.item_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.item, args.latdim)))

        # 定义 GNN 层
        self.num_gnn_layers = args.num_gnn_layers
        self.gnn_layers = nn.ModuleList([GNNLayer() for _ in range(self.num_gnn_layers)])

        # 定义 Mamba 层
        self.num_mamba_layers = args.num_mamba_layers
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=args.latdim,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
                dropout=args.dropout
            ) for _ in range(self.num_mamba_layers)
        ])

        # 自监督学习参数
        self.layer_cl = args.layer_cl
        self.cl_rate = args.lambda_cl
        self.temp = args.tau
        self.eps = args.eps

        # 正则化参数
        self.reg = args.reg

    def build_adj_matrix(self, adj):
        # 优化后的邻接矩阵归一化
        rowsum = np.array(adj.sum(1)).flatten()
        # 添加1e-7避免除零，并计算度数的-0.5次方
        dInvSqrt = np.power(rowsum + 1e-7, -0.5)
        # 将无穷大和NaN的值设置为0
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrt[np.isnan(dInvSqrt)] = 0.0

        # 利用元素级乘法代替矩阵乘法
        adj = adj.tocoo()
        adj.data = dInvSqrt[adj.row] * adj.data * dInvSqrt[adj.col]
        normalized_adj = adj.tocsr()
        return normalized_adj

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        # 将 scipy.sparse 矩阵转换为 torch 稀疏张量
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        ).to(args.device)
        values = torch.from_numpy(sparse_mx.data).to(args.device)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32, device=args.device)

    def forward(self, adj, perturbed=False):
        # 初始化嵌入
        embeds = torch.cat([self.user_embedding, self.item_embedding], dim=0)  # (user + item, latdim)

        # GNN 层的前向传播
        cl_embeds = None
        #layer_embeds = []
        for layer_idx in range(self.num_gnn_layers):
            embeds = self.gnn_layers[layer_idx](adj, embeds)
            embeds = F.normalize(embeds, p=2, dim=1)
            #embeds = embeds * self.scale + self.shift
            #layer_embeds.append(embeds)

            # 保存用于对比学习的嵌入
            if perturbed and self.layer_cl == layer_idx + 1:
                cl_embeds = embeds.clone()
        #embeds = torch.mean(torch.stack(layer_embeds), dim=0)
        # Mamba 层的前向传播
        embeds = embeds.unsqueeze(0)  # 添加 batch 维度
        for layer_idx in range(self.num_mamba_layers):
            embeds = self.mamba_layers[layer_idx](embeds)
            #layer_embeds.append(embeds)

            # 保存用于对比学习的嵌入
            if perturbed and self.layer_cl == self.num_gnn_layers + layer_idx + 1:
                cl_embeds = embeds.squeeze(0).clone()
        #embeds = torch.mean(torch.stack(layer_embeds), dim=0)
        embeds = embeds.squeeze(0)

        user_embeds = embeds[:args.user, :]  # (user, latdim)
        item_embeds = embeds[args.user:, :]  # (item, latdim)

        if perturbed and cl_embeds is not None:
            cl_user_embeds = cl_embeds[:args.user, :]
            cl_item_embeds = cl_embeds[args.user:, :]
            return user_embeds, item_embeds, cl_user_embeds, cl_item_embeds

        return user_embeds, item_embeds

    def calculate_loss(self, user_emb, pos_emb, neg_emb, cl_user_emb=None, cl_item_emb=None):
        # 推荐任务的 BPR 损失
        rec_loss = bpr_loss(user_emb, pos_emb, neg_emb)

        # 正则化损失
        reg_loss = l2_reg_loss(self.reg, user_emb, pos_emb, neg_emb)

        # 对比学习损失
        if cl_user_emb is not None and cl_item_emb is not None:
            cl_loss_user = InfoNCE(user_emb, cl_user_emb, self.temp)
            cl_loss_item = InfoNCE(pos_emb, cl_item_emb, self.temp)
            cl_loss = self.cl_rate * (cl_loss_user + cl_loss_item)
            total_loss = rec_loss + reg_loss + cl_loss
        else:
            total_loss = rec_loss + reg_loss
            cl_loss = torch.tensor(0.0).to(user_emb.device)

        return total_loss, rec_loss, cl_loss

    def calcLosses(self, ancs, poss, negs, adj):
        # 前向传播，获取嵌入
        outputs = self.forward(adj, perturbed=True)
        if len(outputs) == 4:
            user_emb, item_emb, cl_user_emb, cl_item_emb = outputs
        else:
            user_emb, item_emb = outputs
            cl_user_emb, cl_item_emb = None, None

        # 获取当前批次的嵌入
        user_emb_batch = user_emb[ancs]
        pos_emb_batch = item_emb[poss]
        neg_emb_batch = item_emb[negs]

        if cl_user_emb is not None and cl_item_emb is not None:
            cl_user_emb_batch = cl_user_emb[ancs]
            cl_item_emb_batch = cl_item_emb[poss]
        else:
            cl_user_emb_batch, cl_item_emb_batch = None, None

        # 计算损失
        total_loss, rec_loss, cl_loss = self.calculate_loss(
            user_emb_batch, pos_emb_batch, neg_emb_batch, cl_user_emb_batch, cl_item_emb_batch
        )

        return total_loss

    def predict_embeddings(self, adj):
        user_embeds, item_embeds = self.forward(adj)
        return user_embeds, item_embeds

    def predict_score(self, user, item, adj):
        self.eval()
        with torch.no_grad():
            user_embeds, item_embeds = self.forward(adj)
            user_emb = user_embeds[user]
            item_emb = item_embeds[item]
            score = torch.matmul(user_emb, item_emb.transpose(0, 1))
            return score.cpu().numpy()

    def train_model(self, data_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        adj = self.sparse_mx_to_torch_sparse_tensor(self.build_adj_matrix(data_loader.adj)).to(args.device)
        self.to(args.device)

        for epoch in range(args.epochs):
            self.train()
            total_loss = 0
            for batch_idx, (user, pos, neg) in enumerate(data_loader):
                user = user.to(args.device)
                pos = pos.to(args.device)
                neg = neg.to(args.device)

                optimizer.zero_grad()

                # 计算损失
                loss = self.calcLosses(user, pos, neg, adj)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Total Loss: {total_loss / (batch_idx + 1):.4f}')

            # 每个 epoch 后评估模型性能
            self.evaluate(data_loader.validation_set)

    def evaluate(self, validation_set):
        self.eval()
        with torch.no_grad():
            # 实现您的评估逻辑，例如计算准确率、召回率等
            pass

    def predict(self, adj):
        user_embeds, item_embeds = self.forward(adj)
        return user_embeds, item_embeds
