"""Plain UniGCN with the exact structural aggregation used by UniGCNII.

The shared UniGCNIIConv is evaluated with alpha=0 and beta=1, giving W(PX).
There is no restart, identity mixing, or row normalization in this baseline.
Encoder, classifier, and dropout placement match the constrained II comparator.
"""
import torch
import torch.nn.functional as F
import torch_scatter

from .unigcn2 import UniGCNII
from .energy_trace import emit


class UniGCN(UniGCNII):
    def __init__(self, num_features, num_targets, args):
        if args.use_norm:
            raise ValueError('Plain UniGCN requires use_norm=False')
        super().__init__(num_features, num_targets, args)

    def forward(self, data, trace=None):
        x = data.x
        V, E = data.hyperedge_index
        n = x.shape[0]
        m = int(E.max()) + 1
        degrees = torch_scatter.scatter_add(x.new_ones(V.shape[0]), V,
                                           dim=0, dim_size=n).view(-1, 1)
        degE = torch_scatter.scatter(degrees[V], E, dim=0, dim_size=m,
                                    reduce='mean').pow(-0.5)
        degV = degrees.pow(-0.5)
        degE[torch.isinf(degE)] = 1
        degV[torch.isinf(degV)] = 1
        x = F.relu(self.convs[0](self.dropout(x)))
        x0 = x
        emit(trace, 0, 'initial', x)
        e = None
        for i, conv in enumerate(self.convs[1:-1]):
            x, e = conv(self.dropout(x), V, E, 0., 1., x0, degV, degE,
                        trace=trace, step=i + 1)
            x = F.relu(x)
            emit(trace, i + 1, 'hidden_post_activation', x)
        x = self.convs[-1](self.dropout(x))
        emit(trace, self.num_layers, 'logits', x)
        return x, e
