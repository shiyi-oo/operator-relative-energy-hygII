import copy
import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F
from torch.nn import Linear
from torch.autograd import Variable
from lib_models.HNN.utils import SparseMM
from lib_models.HNN.preprocessing import get_HyperGCN_He_dict
from lib_models.HNN.utils import symnormalise,ssm2tst
from collections import defaultdict
from tqdm import tqdm
import math

class HyperGCNConv(nn.Module):
    """
    HyperGCN propagation in the II form, on the LAZY mediator operator.

    The released HyperGCN operator A = D^-1/2 (A_med + I) D^-1/2 is symmetric but has
    NEGATIVE eigenvalues (measured lambda_min ~ -0.24 on real mediator graphs, and
    exactly -2/3 on K_{5,5}). The II energy floor needs spec(P) in [0,1]: on a signed
    spectrum the transient passes near zero and the floor fails -- on K_{5,5} at
    alpha=0.1 the energy dips to 1.1e-4, far below alpha^2 = 1e-2.

    The lazy operator P = (I + A)/2 has spectrum in [0,1] and restores the floor, so it
    is what this layer uses (lazy=True). Set lazy=False for the raw operator.
    """
    def __init__(self, a, b, reapproximate=True, lazy=True, zero_bias=True):
        super(HyperGCNConv, self).__init__()
        self.a, self.b = a, b
        self.reapproximate = reapproximate
        self.lazy = lazy
        self.zero_bias = zero_bias

        self.W = nn.Parameter(torch.FloatTensor(a, b))
        self.bias = nn.Parameter(torch.FloatTensor(b))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        # A nonzero constant bias sits outside ker E_P = span(D^1/2 1) and injects energy
        # at every layer, which masks over-smoothing in the diagnostic. Zero init matches
        # every other model in this package (hgnn.py uses zeros(bias)).
        if self.zero_bias:
            self.bias.data.zero_()
        else:
            self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, m=True, alpha=None, beta=None, H0=None):
#         ipdb.set_trace()
        W, b = self.W, self.bias

        if self.reapproximate:
            # HW is used ONLY to choose the (supremum, infimum) pair per hyperedge.
            # The II layer propagates H itself, so in the fast path (reapproximate=False)
            # computing HW here would be a wasted dense matmul every layer.
            HW = torch.mm(H, W)
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else:
            A = structure

        A = A.to(H.device)
        A = Variable(A)

        AH = SparseMM.apply(A, H)
        # lazy operator: P H = (I + A) H / 2, so that spec(P) in [0,1]
        PH = 0.5 * (H + AH) if self.lazy else AH

        # X^(l+1) = sigma( ((1-alpha) P X^(l) + alpha X^(0)) W_beta ), W_beta=(1-beta)I+beta W
        Hi = (1 - alpha) * PH + alpha * H0
        H = (1 - beta) * Hi + beta * Hi @ W

        return H + b

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'

class HyperGCNII(nn.Module):
    def __init__(self, num_features, num_targets, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperGCNII, self).__init__()

        self.fast = args.HyperGCN_fast
        self.structure, self.mediator = None, args.HyperGCN_mediators
        self.dropout, self.num_layers = args.dropout, args.All_num_layers
        self.hidden_dim = args.MLP_hidden
        self.lamda = args.lamda
        self.alpha = args.restart_alpha

        self.lazy = bool(getattr(args, 'HyperGCN_lazy', True))
        zero_bias = bool(getattr(args, 'HyperGCN_zero_bias', True))

        self.convs = nn.ModuleList()
        self.convs.append(Linear(num_features, self.hidden_dim))
        for _ in range(self.num_layers):
            self.convs.append(HyperGCNConv(
                self.hidden_dim, self.hidden_dim, (not self.fast),
                lazy=self.lazy, zero_bias=zero_bias))
        self.convs.append(Linear(self.hidden_dim, num_targets))

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

        lamda, alpha = self.lamda, self.alpha
        H = data.x
        H = F.relu(self.convs[0](H))
        H0 = H
        for i, conv in enumerate(self.convs[1:-1]):
            beta = math.log(lamda/(i+1)+1)
            H = F.relu(conv(self.structure, H, self.mediator, alpha, beta, H0))
            if i < self.num_layers - 1:
                H = F.dropout(H, self.dropout, training=self.training)
        H = self.convs[-1](H)

        return H, None 

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
