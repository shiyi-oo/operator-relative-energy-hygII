"""Sparse, float64 energy diagnostics on the largest incidence component.

No clique expansion is materialized. Energies are evaluated using centered
within-edge variances, including near the Perron kernel where trace subtraction
is inaccurate. This module has no dependency on training code or torch.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigsh


def array_hash(x):
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(str(x.dtype).encode())
    h.update(x.tobytes())
    return h.hexdigest()


@dataclass
class Incidence:
    H: sparse.csr_matrix
    nodes: np.ndarray
    full_nodes: int
    component_sizes: list
    source_hash: str

    @classmethod
    def largest_component(cls, index, n):
        index = np.asarray(index, dtype=np.int64)
        if index.ndim != 2 or index.shape[0] != 2 or index.shape[1] == 0:
            raise ValueError('Expected a nonempty 2 by nnz incidence index')
        v, e = index
        if v.min() < 0 or v.max() >= n or e.min() < 0:
            raise ValueError('Incidence index out of bounds')
        m = int(e.max()) + 1
        H = sparse.coo_matrix((np.ones(len(v)), (v, e)), shape=(n, m)).tocsr()
        if H.nnz != len(v) or not np.all(H.data == 1):
            raise ValueError('Duplicate incidence memberships')
        if np.any(np.asarray(H.sum(0)).ravel() == 0):
            raise ValueError('Empty/noncontiguous hyperedges')
        if np.any(np.asarray(H.sum(1)).ravel() == 0):
            raise ValueError('Isolated nodes must have processed singleton edges')
        graph = sparse.bmat([[None, H], [H.T, None]], format='csr')
        _, labels = connected_components(graph, directed=False)
        vertex_labels = labels[:n]
        groups = [np.flatnonzero(vertex_labels == c) for c in np.unique(vertex_labels)]
        groups.sort(key=lambda g: (-len(g), int(g[0])))
        nodes = groups[0]
        sub = H[nodes]
        sub = sub[:, np.flatnonzero(np.asarray(sub.sum(0)).ravel())].tocsr()
        return cls(sub, nodes, n, [len(g) for g in groups], array_hash(index))


class OperatorEnergy:
    """P = diag(b) H diag(k) H.T diag(c), on one connected component."""

    def __init__(self, incidence, family, hnhn_alpha=-1.5, hnhn_beta=-0.5,
                 chunk_size=32768):
        self.incidence = incidence
        self.H = incidence.H
        self.family = family
        self.chunk_size = chunk_size
        n, m = self.H.shape
        self.d = np.asarray(self.H.sum(1)).ravel()
        self.card = np.asarray(self.H.sum(0)).ravel()
        self.b = np.ones(n)
        self.c = np.ones(n)
        if family == 'HGNN':
            self.b = self.d ** -0.5
            self.c = self.b.copy()
            self.k = 1 / self.card
            psi = np.sqrt(self.d)
        elif family == 'HNHN':
            self.c = self.d ** hnhn_beta
            edge_weight = self.card ** hnhn_alpha
            self.b = 1 / (self.H @ edge_weight)
            self.k = edge_weight / (self.H.T @ self.c)
            psi = np.ones(n)
        elif family == 'UniGCN':
            self.b = self.d ** -0.5
            mean_degree = (self.H.T @ self.d) / self.card
            self.k = 1 / (self.card * np.sqrt(mean_degree))
            psi = None
        else:
            raise ValueError(f'Unsupported family {family}')
        self.pi_scale = 1 / np.mean(self.c / self.b)
        self.pi = (self.c / self.b) * self.pi_scale
        root_pi = np.sqrt(self.pi)
        def symmetric(x):
            return root_pi * self.apply(x / root_pi)
        self.symmetric = LinearOperator((n, n), matvec=symmetric, dtype=np.float64)
        if n <= 256:
            # Small numerical fixtures only; real datasets remain matrix-free.
            mat = self.apply(np.eye(n)) * root_pi[:, None] / root_pi[None, :]
            values, vectors = np.linalg.eigh(mat)
            self.lambda1 = float(values[-1])
            self.lambda2 = float(values[-2]) if n > 1 else None
            self.lambda_min = float(values[0])
            computed = vectors[:, -1] / root_pi
        else:
            values, vectors = eigsh(self.symmetric, k=2, which='LA', tol=1e-12,
                                   maxiter=100000, v0=np.ones(n))
            order = np.argsort(values)
            self.lambda1 = float(values[order[-1]])
            self.lambda2 = float(values[order[-2]])
            # PSD is structural; no expensive smallest-eigenvalue solve is needed.
            self.lambda_min = None
            computed = vectors[:, order[-1]] / root_pi
        if psi is None:
            psi = computed * (1 if computed.sum() >= 0 else -1)
        else:
            self.lambda1 = 1.0
        if np.any(psi <= 0) or self.lambda1 <= 0:
            raise ValueError('Perron vector is not strictly positive')
        self.psi = psi / np.sqrt(np.sum(self.pi * psi ** 2))
        residual = self.apply(self.psi) - self.lambda1 * self.psi
        self.residual = float(np.sqrt(np.sum(self.pi * residual ** 2)) / self.lambda1)
        if self.residual > 1e-10:
            raise ValueError(f'Uncertified Perron pair: relative residual {self.residual}')
        self.v, self.e = self.H.nonzero()
        self.z = self.c * self.psi
        self.t = self.H.T @ self.z
        self.perron_min = float(self.psi.min())

    def apply(self, x):
        if x.ndim == 1:
            return self.b * (self.H @ (self.k * (self.H.T @ (self.c * x))))
        return self.b[:, None] * (self.H @ (self.k[:, None] * (self.H.T @ (self.c[:, None] * x))))

    def variance(self, x, vertex_weights, edge_weights):
        """Sum_e edge_weight_e sum_v vertex_weight_v ||x_v-mean_e||^2."""
        mass = self.H.T @ vertex_weights
        means = (self.H.T @ (vertex_weights[:, None] * x)) / mass[:, None]
        result = 0.0
        for start in range(0, len(self.v), self.chunk_size):
            v = self.v[start:start + self.chunk_size]
            e = self.e[start:start + self.chunk_size]
            diff = x[v] - means[e]
            squares = np.einsum('ij,ij->i', diff, diff)
            result += float(np.dot(edge_weights[e] * vertex_weights[v], squares))
        return result

    def metrics(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] != self.H.shape[0] or not np.isfinite(x).all():
            raise ValueError('Features must be finite and match the measured component')
        # A temporary global rescaling improves quotient accuracy without altering
        # the forward trajectory. Raw values are restored using the same scale.
        scale = float(np.max(np.abs(x)))
        if scale == 0:
            return dict(energy_un=0., energy_sym=0., energy_op=0., norm_f2=0.,
                        norm_pi2=0., R_un=None, R_sym=None, R_op=None,
                        R_op_euclidean=None, delta_op2=None, delta_consensus2=None,
                        zero_state=True, zero_rows=len(x))
        y = x / scale
        norm_f = float(np.sum(y * y))
        norm_pi = float(np.sum(self.pi[:, None] * y * y))
        un = self.variance(y, np.ones(len(y)), np.ones(self.H.shape[1]))
        sym = self.variance(y / np.sqrt(self.d[:, None]), np.ones(len(y)),
                            np.ones(self.H.shape[1]))
        op = self.variance(y / self.psi[:, None], self.z,
                           self.k * self.t / self.lambda1) * self.pi_scale
        projection = self.psi[:, None] * (self.psi @ (self.pi[:, None] * y))[None, :]
        delta = float(np.sum(self.pi[:, None] * (y - projection) ** 2) / norm_pi)
        consensus = float(np.sum((y - y.mean(0)) ** 2) / norm_f)
        scale2 = scale * scale
        return dict(energy_un=un * scale2, energy_sym=sym * scale2,
                    energy_op=op * scale2, norm_f2=norm_f * scale2,
                    norm_pi2=norm_pi * scale2, R_un=un / norm_f, R_sym=sym / norm_f,
                    R_op=op / norm_pi, R_op_euclidean=op / norm_f,
                    delta_op2=delta, delta_consensus2=consensus,
                    zero_state=False, zero_rows=int(np.sum(np.all(x == 0, axis=1))))

    def metadata(self):
        def angle(direction):
            direction = direction / np.sqrt(np.sum(self.pi * direction ** 2))
            return float(np.degrees(np.arccos(np.clip(np.sum(self.pi * self.psi * direction), -1, 1))))
        kernel = self.metrics(self.psi[:, None])
        return dict(family=self.family, nodes=self.H.shape[0], edges=self.H.shape[1],
                    incidence_nnz=self.H.nnz, full_nodes=self.incidence.full_nodes,
                    component_sizes=self.incidence.component_sizes,
                    node_fraction=len(self.incidence.nodes) / self.incidence.full_nodes,
                    incidence_hash=self.incidence.source_hash, lambda1=self.lambda1,
                    lambda2=self.lambda2, lambda_min=self.lambda_min,
                    perron_residual=self.residual, perron_min=self.perron_min,
                    pi_mean=float(self.pi.mean()), angle_constant=angle(np.ones(len(self.d))),
                    angle_sqrt_degree=angle(np.sqrt(self.d)),
                    kernel_control_R_op=kernel['R_op'])

