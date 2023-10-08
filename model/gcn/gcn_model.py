import torch.nn as nn
import torch.nn.functional as F
import torch
class GCN(nn.Module):

    def __init__(self, emb_dim=768, num_layers=2, gcn_dropout=0.5):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)


    def forward(self, X, adj):
        # gcn layer
        degree = adj.sum(2).unsqueeze(2) + 1     # 度矩阵 [batch * maxlen * 1]
        # mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)

        for layer in range(self.layers):
            # ax = adj.bmm(X)  # 1,100,100  1,100,768 : 计算A * X
            ax = torch.bmm(adj, X)

            axW = self.W[layer](ax)
            axW = axW + self.W[layer](X)  # self loop
            # axW = axW.cuda() / denom
            axW = axW / degree       # 归一化，除以D==乘以D-1
            gaxW = F.gelu(axW)
            X = self.gcn_drop(gaxW) if layer < self.layers - 1 else gaxW    # 判断是否需要drop
        return X

    #            print('Ax.is_cuda:', Ax.is_cuda)
    #            print(Ax)
    #            print(self.W[0])
    #            print(self.W[0](Ax.cpu()))

