"""Matched HNHN baseline and II used in the main classification study.

Both arms use the same encoder, L square two-stage layers, and classifier.
Each stage retains a trainable bias. Identity mixing acts on the weights;
biases stay additive in both arms (this is not a bias-free theorem experiment).
"""
import math

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing

from .energy_trace import emit


class MatchedHNHNConv(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='add', node_dim=0)
        self.weight_v2e = nn.Linear(channels, channels, bias=True)
        self.weight_e2v = nn.Linear(channels, channels, bias=True)

    def reset_parameters(self):
        self.weight_v2e.reset_parameters()
        self.weight_e2v.reset_parameters()

    @staticmethod
    def channel(x, linear, beta):
        # Preserve bias presence, position, initialization, and parameter count.
        return (1 - beta) * x + beta * F.linear(x, linear.weight) + linear.bias

    def forward(self, x, data, alpha, beta, x0):
        index = data.hyperedge_index
        n = x.size(0)
        m = int(index[1].max()) + 1 if index.numel() else 0
        x = self.channel(x, self.weight_v2e, beta)
        self.flow = 'source_to_target'
        edge_pre = self.propagate(
            index, x=data.D_v_beta[:, None] * x,
            norm=data.D_e_beta_inv, size=(n, m))
        edge = F.relu(edge_pre)
        self.flow = 'target_to_source'
        node = self.propagate(
            index, x=data.D_e_alpha[:, None] * edge,
            norm=data.D_v_alpha_inv, size=(n, m))
        node = (1 - alpha) * node + alpha * x0
        # Moving the second affine map after B is equivalent to native HNHN
        # when alpha=0, beta=1: B is row-stochastic, including singleton nodes.
        return self.channel(node, self.weight_e2v, beta), edge_pre

    def message(self, x_j, norm_i):
        return norm_i[:, None] * x_j


class MatchedHNHN(nn.Module):
    def __init__(self, num_features, num_targets, args, *, use_ii=False):
        super().__init__()
        if args.All_num_layers < 1:
            raise ValueError('Matched HNHN requires at least one propagation layer')
        self.use_ii = use_ii
        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        self.alpha = args.restart_alpha if use_ii else 0.
        self.lamda = args.lamda
        self.encoder = nn.Linear(num_features, args.MLP_hidden)
        self.convs = nn.ModuleList(
            MatchedHNHNConv(args.MLP_hidden) for _ in range(self.num_layers))
        self.classifier = nn.Linear(args.MLP_hidden, num_targets)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, data, trace=None):
        x = F.relu(self.encoder(data.x))
        x0 = x
        emit(trace, 0, 'initial', x0)
        edge = None
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            beta = math.log1p(self.lamda / (i + 1)) if self.use_ii else 1.
            x, edge = conv(x, data, self.alpha, beta, x0)
            x = F.relu(x)
            emit(trace, i + 1, 'hidden_post_activation', x)
        x = self.classifier(F.dropout(x, p=self.dropout, training=self.training))
        emit(trace, self.num_layers, 'logits', x)
        return x, edge
