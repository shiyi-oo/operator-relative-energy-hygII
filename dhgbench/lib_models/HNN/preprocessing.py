"""Model-specific preprocessing used by HNHN and HyperGCN."""
import copy
from collections import defaultdict
import numpy as np
import torch
import torch_scatter


def algo_preprocessing(data, args):
    if args.method in ('HNHN', 'HNHNII'):
        data = generate_HNHN_norm(data, args)
    return data


def generate_HNHN_norm(data, args):
    """
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """

    edge_index=copy.deepcopy(data.hyperedge_index)
    edge_index = edge_index.to('cpu')

    # Construct incidence matrix H of size (num_nodes, num_hyperedges) from edge_index = [V;E]
    ones = torch.ones(edge_index.shape[1], device=edge_index.device)

    alpha = args.HNHN_alpha
    beta = args.HNHN_beta

    # the degree of the node
    DV = torch_scatter.scatter_add(ones, edge_index[0], dim=0)
    # the degree of the hyperedge
    DE = torch_scatter.scatter_add(ones, edge_index[1], dim=0)

    # alpha part
    D_e_alpha = DE ** alpha
    D_e_alpha[D_e_alpha == float("inf")] = 0 
    D_v_alpha = torch_scatter.scatter_add(D_e_alpha[edge_index[1]], edge_index[0], dim=0)

    # beta part
    D_v_beta = DV ** beta
    D_v_beta[D_v_beta == float("inf")] = 0 
    D_e_beta = torch_scatter.scatter_add(D_v_beta[edge_index[0]], edge_index[1], dim=0)

    D_v_alpha_inv = 1.0 / D_v_alpha
    D_v_alpha_inv[D_v_alpha_inv == float("inf")] = 0

    D_e_beta_inv = 1.0 / D_e_beta
    D_e_beta_inv[D_e_beta_inv == float("inf")] = 0

    data.D_e_alpha,data.D_v_alpha_inv,data.D_v_beta,data.D_e_beta_inv = D_e_alpha.float().to(args.device),D_v_alpha_inv.float().to(args.device),D_v_beta.float().to(args.device),D_e_beta_inv.float().to(args.device)
    
    return data


def get_HyperGCN_He_dict(data):
    edge_index = np.array(data.hyperedge_index.cpu())
    He_dict = defaultdict(list)

    for node, he in zip(edge_index[0, :], edge_index[1, :]):
        He_dict[he.item()].append(node)

    return dict(He_dict)
