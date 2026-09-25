import copy
import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
from lib_models.HNN.utils import SparseMM
from lib_models.HNN.preprocessing import get_HyperGCN_He_dict
from lib_models.HNN.utils import symnormalise,ssm2tst
from collections import defaultdict
from tqdm import tqdm

class HyperGCNConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, a, b, reapproximate=True, zero_bias=True):
        super(HyperGCNConv, self).__init__()
        self.a, self.b = a, b
        self.reapproximate = reapproximate
        self.zero_bias = zero_bias

        self.W = nn.Parameter(torch.FloatTensor(a, b))
        self.bias = nn.Parameter(torch.FloatTensor(b))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        # A nonzero constant bias sits outside ker E_P = span(D^1/2 1) and injects energy
        # at every layer: at init it makes plain HyperGCN look like it does NOT over-smooth
        # (relative Rayleigh quotient plateaus at ~2.8e-2 instead of decaying to ~2e-10).
        # Zero init matches every other model here (hgnn.py uses zeros(bias)).
        # NOTE the bias is still trainable and is NOT inside W_beta, so a TRAINED model can
        # re-acquire a constant component; measure at init, or report that caveat.
        if self.zero_bias:
            self.bias.data.zero_()
        else:
            self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m=True):
#         ipdb.set_trace()
        W, b = self.W, self.bias
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else: A = structure

        A = A.to(H.device)
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'

class HyperGCN(nn.Module):
    def __init__(self, num_features, num_targets, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCN, self).__init__()

        self.fast = args.HyperGCN_fast
        self.structure, self.mediator = None, args.HyperGCN_mediators
        self.dropout, self.num_layers = args.dropout, args.All_num_layers
        self.hidden_dim = args.MLP_hidden

        zb = bool(getattr(args, 'HyperGCN_zero_bias', True))

        self.convs = nn.ModuleList()
        if self.num_layers == 1:
            self.convs.append(HyperGCNConv(
                num_features, num_targets, (not self.fast), zero_bias=zb))
        else:
            self.convs.append(HyperGCNConv(
                num_features, self.hidden_dim, (not self.fast), zero_bias=zb))
            for _ in range(self.num_layers-2):
                self.convs.append(HyperGCNConv(
                self.hidden_dim, self.hidden_dim, (not self.fast), zero_bias=zb))
            self.convs.append(HyperGCNConv(
                self.hidden_dim, num_targets, (not self.fast), zero_bias=zb))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        if self.structure is None:
            # print('Precomputing ...')
            data_copy = copy.deepcopy(data)
            data_copy.to('cpu')
            He_dict = get_HyperGCN_He_dict(data_copy) 
            
            if self.fast:
                self.structure = Laplacian(V=data_copy.x.shape[0], E=He_dict, X=data_copy.x, m=self.mediator)
            else:
                self.structure = He_dict

        H = data.x
        
        for i, conv in enumerate(self.convs):
            H = F.relu(conv(self.structure, H, self.mediator))
            if i < self.num_layers - 1:
                H = F.dropout(H, self.dropout, training=self.training)

        return H,None 

def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    edges = [list(itm) for itm in dictionary.keys()]   
    organised = []

    for e in edges:
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)

    edges, weights = np.array(edges), np.array(organised)
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A

def Laplacian(V, E, X, m):
    """
    Approximates hypergraph Laplacian using clique expansion or mediator method.

    Args:
        V (int): number of vertices
        E (dict): hyperedge dictionary {hid: list of nodes}
        X (ndarray): node features
        m (bool): use mediator mode or not

    Returns:
        A (torch.sparse.Tensor): normalized adjacency matrix
    """
    edges = []
    weights = defaultdict(float)
    rv = np.random.rand(X.shape[1])
    X_proj = X @ rv  # shape: (n_nodes,)

    for hyperedge in tqdm(E.values(), disable=True):

        hyperedge = list(hyperedge)
        proj = X_proj[hyperedge]
        s, i = np.argmax(proj), np.argmin(proj)
        Se, Ie = hyperedge[s], hyperedge[i]

        if m:
            c = max(2 * len(hyperedge) - 3, 1)

            edges.extend([[Se, Ie], [Ie, Se]])
            weights[(Se, Ie)] += 1.0 / c
            weights[(Ie, Se)] += 1.0 / c

            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    for u, v in [(Se, mediator), (Ie, mediator), (mediator, Se), (mediator, Ie)]:
                        edges.append([u, v])
                        weights[(u, v)] += 1.0 / c
        else:
            e = len(hyperedge)
            edges.extend([[Se, Ie], [Ie, Se]])
            weights[(Se, Ie)] += 1.0 / e
            weights[(Ie, Se)] += 1.0 / e

    return adjacency(edges, weights, V)
