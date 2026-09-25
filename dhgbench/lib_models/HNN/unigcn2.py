import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch_scatter
from lib_models.HNN.utils import normalize_l2
from lib_models.HNN.energy_trace import emit

class UniGCNIIConv(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, X, vertex, edges, alpha, beta, X0, degV, degE, trace=None, step=0):
        N = X.shape[-2]

        Xve = X[..., vertex, :] # [nnz, C]
        Xe = torch_scatter.scatter(Xve, edges, dim=-2, reduce="mean") # [E, C], reduce is 'mean' here as default
        Xe = Xe * degE 
        Xev = Xe[..., edges, :] # [nnz, C]
        Xv = torch_scatter.scatter(Xev, vertex, dim=-2, reduce="sum", dim_size=N) # [N, C]
        Xv = Xv * degV
        X = Xv
        emit(trace, step, 'aggregation_pre_norm', X)

        if self.args.use_norm:
            X = normalize_l2(X)
            emit(trace, step, 'post_row_norm', X)

        Xi = (1-alpha) * X + alpha * X0
        X = (1-beta) * Xi + beta * self.W(Xi)

        return X, Xe

    def flops(self, X, vertex, edges):
        flops = vertex.shape[0] + edges.shape[0] + edges.shape[0] + vertex.shape[0] # scatter
        flops += np.prod(X.shape) # init connection
        flops += np.prod(X.shape[:-1]) * self.in_features * self.out_features # linear
        return flops


class UniGCNII(nn.Module):
    def __init__(self, num_features, num_targets, args):

        super().__init__()
        self.num_layers = args.All_num_layers
        self.nhid = args.MLP_hidden

        self.in_features = num_features
        self.out_features = num_targets

        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation] # Default relu
        self.input_drop = nn.Dropout(args.input_drop) # 0.6 is chosen as default
        self.dropout = nn.Dropout(args.dropout) # 0.2 is chosen for GCNII
        self.alpha = args.restart_alpha
        self.lamda = args.lamda

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(num_features, self.nhid))
        for _ in range(self.num_layers):
            self.convs.append(UniGCNIIConv(args, self.nhid, self.nhid))
        self.convs.append(torch.nn.Linear(self.nhid, num_targets))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        
    def forward(self, data, trace=None):
        x = data.x
        V, E = data.hyperedge_index[0], data.hyperedge_index[1]
        num_nodes = x.shape[0]
        num_edges = int(E.max()) + 1 if E.numel() > 0 else 0

        degV = torch_scatter.scatter_add(torch.ones(V.shape[0], device=x.device, dtype=x.dtype),V,dim=0,dim_size=num_nodes,).view(-1, 1)
        degE = torch_scatter.scatter(degV[V],E,dim=0,reduce='mean',dim_size=num_edges)
        degE = degE.pow(-0.5)
        degE[torch.isinf(degE)] = 1
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1

        lamda, alpha = self.lamda, self.alpha
        x = self.dropout(x)
        x = F.relu(self.convs[0](x)) 
        x0 = x
        emit(trace, 0, 'initial', x0)
        for i,con in enumerate(self.convs[1:-1]):
            x = self.dropout(x)
            beta = math.log(lamda/(i+1)+1)
            x,e = con(x, V, E, alpha, beta, x0, degV, degE, trace=trace, step=i + 1)
            x = F.relu(x)
            emit(trace, i + 1, 'hidden_post_activation', x)
        x = self.dropout(x)
        x = self.convs[-1](x)
        emit(trace, self.num_layers, 'logits', x)
        return x,e

    def flops(self, data):
        x = data.x
        V, E = data.edge_index[0], data.edge_index[1]
        flops = self.in_features * self.nhid * np.prod(x.shape[:-1]) # linear
        flops += self.nhid * np.prod(x.shape[:-1]) # non-linear
        for i,con in enumerate(self.convs[1:-1]):
            flops += con.flops(x, V, E) # conv
            flops += self.nhid * np.prod(x.shape[:-1]) # non-linear
        flops = self.out_features * self.nhid * np.prod(x.shape[:-1]) # linear
        return flops
