import torch
import pickle
import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_sparse import coalesce

def load_dataset(path='../hyperGCN/data/', dataset = 'cora', train_percent = 0.025):
    
    # first load node features:
    with open(osp.join(path, dataset, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        # cocitation/coauthorship pickle a scipy sparse matrix; the heterophilous and
        # the fair datasets pickle a dense array already. Either way torch.FloatTensor
        # below needs it dense.
        if hasattr(features, 'todense'):
            features = features.todense()

    # then load node labels:
    with open(osp.join(path, dataset, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(osp.join(path, dataset, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = int)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    n_x = num_nodes
    data.n_x = n_x

    data.train_percent = train_percent
    data.num_hyperedges = len(hypergraph)
    
    return data


def load_cornell_dataset(path='../data/raw_data/', dataset = 'house-committees', 
        feature_noise = 0.1,
        feature_dim = None,
        train_percent = 0.025):

    # first load node labels
    df_labels = pd.read_csv(osp.join(path, dataset, f'node-labels-{dataset}.txt'), names = ['node_label'])
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    p2hyperedge_list = osp.join(path, dataset, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list, 
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    n_x = num_nodes
    data.n_x = n_x
    
    data.train_percent = train_percent
    data.num_hyperedges = he_id - num_nodes
    
    return data


def load_LE_dataset(path=None, dataset="NTU2012", train_percent = 0.025):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))
    
    file_name = f'{dataset}.content'
    p2idx_features_labels = osp.join(path, dataset, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))

    print ('load features')
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'
    p2edges_unordered = osp.join(path, dataset, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')

    # From adjacency matrix to edge_list
    edge_index = edges.T 
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    
    # build torch data class
    data = Data(
            x = torch.FloatTensor(np.array(features[:num_nodes].todense())), 
            edge_index = torch.LongTensor(edge_index),
            y = labels[:num_nodes])

    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    n_x = num_nodes
    num_class = len(np.unique(labels[:num_nodes].numpy()))
    data.n_x = n_x
    data.y=data.y-data.y.min()
    
    data.train_percent = train_percent
    data.num_hyperedges = num_he
    
    return data

