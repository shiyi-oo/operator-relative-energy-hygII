import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import math
from lib_models.HNN.energy_trace import emit

class HNHNConv(MessagePassing):
    """HNHN propagation in the II form.

    P = A_VE A_EV is the fixed product-form operator of the HNHN row:
        A_EV = diag(sum_{u in e} d_u^beta)^-1 H^T diag(d^beta)      (vertex -> edge)
        A_VE = diag(sum_{e ni v} |e|^alpha)^-1 H  diag(|e|^alpha)   (edge   -> vertex)
    It is row-stochastic, reversible wrt pi_v = d_v^beta sum_{e ni v}|e|^alpha, and PSD,
    so spec(P) in [0,1] with lambda_1 = 1 and psi = 1.

    The layer is exactly
        X^(l+1) = sigma( ((1-alpha) P X^(l) + alpha X^(0)) W_beta ),
        W_beta  = (1-beta) I + beta W,   beta_l = log(lamda/l + 1).
    """

    def __init__(self, in_channels, hidden_channels, out_channels, nonlinear_inbetween=False,
                 concat=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HNHNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        # The II definition applies sigma AFTER the propagation, so the two weighted-mean
        # stages have to compose into the single fixed linear operator P. An inner
        # nonlinearity breaks that, so it is off by default here; hnhn.py (the base model)
        # keeps HNHN's published inner ReLU.
        self.nonlinear_inbetween = nonlinear_inbetween

        self.concat = True

        # ONE weight, entering only through W_beta = (1-beta)I + beta W. A second,
        # un-damped weight inside the propagation would break the identity mapping:
        # beta_l -> 0 would no longer send the layer to the identity, and the depth-uniform
        # floor would pick up an unconstrained sigma_min(W)^2 factor at every layer.
        assert in_channels == out_channels, \
            'HNHNII needs square layers so that W_beta = (1-beta)I + beta W is well defined'
        self.weight = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.reset_parameters()

    def forward(self, x, data, alpha, beta, X0):

        hyperedge_index = data.hyperedge_index
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        # ---- vertex -> edge:  A_EV = diag(D_e_beta)^-1 H^T diag(d^beta) ----
        x = data.D_v_beta.unsqueeze(-1) * x

        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=data.D_e_beta_inv,
                             size=(num_nodes, num_edges))

        edge_embed = out

        if self.nonlinear_inbetween:
            out = F.relu(out)

        out = torch.squeeze(out, dim=1)

        # ---- edge -> vertex:  A_VE = diag(D_v_alpha)^-1 H diag(|e|^alpha) ----
        out = data.D_e_alpha.unsqueeze(-1) * out

        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=data.D_v_alpha_inv,
                             size=(num_nodes, num_edges))

        # ---- II combination: restart, then the identity-mapped weight ----
        Xi = (1 - alpha) * out + alpha * X0
        out = (1 - beta) * Xi + beta * self.weight(Xi)

        return out, edge_embed

    def message(self, x_j, norm_i):

        out = norm_i.view(-1, 1) * x_j

        return out

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.hidden_channels, self.out_channels)

class HNHNII(nn.Module):

    def __init__(self, num_features, num_targets, args):
        super(HNHNII, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.hidden_dim = args.MLP_hidden

        self.alpha = args.restart_alpha
        self.lamda = args.lamda

        # Off by default (see HNHNConv). Set HNHN_II_nonlinear_inbetween to restore
        # HNHN's published inner ReLU -- outside the scope of the II theory.
        inner_nl = bool(getattr(args, 'HNHN_II_nonlinear_inbetween', False))

        self.convs = nn.ModuleList()
        self.convs.append(torch.nn.Linear(num_features, self.hidden_dim))
        for _ in range(self.num_layers):
            self.convs.append(HNHNConv(self.hidden_dim, self.hidden_dim, self.hidden_dim,
                                       nonlinear_inbetween=inner_nl))
        self.convs.append(torch.nn.Linear(self.hidden_dim, num_targets))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data, trace=None):

        x = data.x
        lamda, alpha = self.lamda, self.alpha
        x = F.relu(self.convs[0](x))
        xv0 = x
        emit(trace, 0, 'initial', xv0)
        e = None
        for i, conv in enumerate(self.convs[1:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            beta = math.log(lamda / (i + 1) + 1)
            x, e = conv(x, data, alpha, beta, xv0)
            x = F.relu(x)
            emit(trace, i + 1, 'hidden_post_activation', x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        emit(trace, self.num_layers, 'logits', x)

        return x, e
