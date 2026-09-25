import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional
from lib_models.HNN.utils import zeros,glorot
from lib_models.HNN.energy_trace import emit
import math

class HypergraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, symdegnorm=False, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.symdegnorm = symdegnorm

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = nn.Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor, alpha2, beta, x0, D, B,
                hyperedge_weight: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        # if hyperedge_weight is None:
        #     hyperedge_weight = x.new_ones(num_edges,device=x.device)

        alpha = None
        if self.use_attention:
            assert num_edges <= num_edges
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if not self.symdegnorm:
            self.flow = 'source_to_target'
            edge_embed = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,size=(num_nodes, num_edges))
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=edge_embed, norm=D, alpha=alpha,size=(num_nodes, num_edges))
            
        else:  # this correspond to HGNN
            x = D.unsqueeze(-1)*x
            V, E = hyperedge_index[0], hyperedge_index[1]
            # node -> hyperedge
            Xve = x[V]                          # [nnz, C]
            # B = 1/|e| already carries the mean over the hyperedge, so aggregate with
            # 'sum' here. Using 'mean' AND multiplying by B applies 1/|e| twice, which
            # makes P = D^-1/2 H B^-2 H^T D^-1/2 and drops lambda_1 from 1 to ~1/|e|.
            Xe = torch_scatter.scatter(Xve, E, dim=0, dim_size=num_edges, reduce='sum')
            edge_embed = Xe * B.unsqueeze(-1)

            # hyperedge -> node
            Xev = edge_embed[E]                 # [nnz, C]
            Xv = torch_scatter.scatter(Xev, V, dim=0, dim_size=num_nodes, reduce='sum')
            out = Xv * D.unsqueeze(-1)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            edge_embed = edge_embed.view(-1,self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            edge_embed = edge_embed.mean(dim=-1)

        Xi = (1 - alpha2) * out + alpha2 * x0
        out = (1 - beta) * Xi + beta * torch.matmul(Xi, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out,edge_embed 

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels
        
        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class HCHAII(nn.Module):

    def __init__(self, num_features, num_targets, args):
        super(HCHAII, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HCHA_symdegnorm
        self.hidden_dim = args.MLP_hidden
        self.alpha2 = args.restart_alpha
        self.lamda2 = args.lamda

        self.lin_in = nn.Linear(num_features, self.hidden_dim)
        self.lin_out = nn.Linear(self.hidden_dim, num_targets)
        self.convs = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(HypergraphConv(
                self.hidden_dim, self.hidden_dim, self.symdegnorm))

    def reset_parameters(self):
        self.lin_in.reset_parameters()
        self.lin_out.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, trace=None):

        # regular node classification

        x = data.x
        num_nodes = x.shape[0]
        num_edges = int(data.hyperedge_index[1].max()) + 1
        lamda2, alpha2 = self.lamda2, self.alpha2
        x = F.elu(self.lin_in(x))
        x0 = x
        emit(trace, 0, 'initial', x0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        e = None

        hyperedge_index = data.hyperedge_index
        # key the cache: D and B depend on the incidence, on symdegnorm, and on dtype/device,
        # so a bare hasattr check would silently reuse stale tensors if any of those change
        key = (int(num_nodes), int(num_edges), int(hyperedge_index.size(1)),
               bool(self.symdegnorm), str(x.device), str(x.dtype))
        if getattr(data, '_DB_key', None) != key:
            data._DB_key = key
            data._DB = get_DB(hyperedge_index, num_nodes, num_edges, x.device,
                              self.symdegnorm, dtype=x.dtype)
        D, B = data._DB

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            beta = math.log(lamda2/(i+1)+1)
            x , e = conv(x, hyperedge_index, alpha2, beta, x0, D, B) 
            x = F.elu(x)
            emit(trace, i + 1, 'hidden_post_activation', x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_out(x)
        emit(trace, self.num_layers, 'logits', x)

        return x,e

    @torch.no_grad()
    def predict(self,data):
        self.eval()
        return self.forward(data)
    
def get_DB(hyperedge_index, num_nodes, num_edges, device, symdegnorm=True, dtype=None):
    dtype = torch.get_default_dtype() if dtype is None else dtype
    one_e = torch.ones(num_edges, device=device, dtype=dtype)
    one_i = torch.ones(hyperedge_index.size(1), device=device, dtype=dtype)

    D = scatter_add(one_e[hyperedge_index[1]], hyperedge_index[0], dim=0, dim_size=num_nodes)
    D = D.pow(-0.5 if symdegnorm else -1.0)
    D[torch.isinf(D)] = 0

    B = scatter_add(one_i, hyperedge_index[1], dim=0, dim_size=num_edges)
    B = B.pow(-1.0)
    B[torch.isinf(B)] = 0
    return D, B
