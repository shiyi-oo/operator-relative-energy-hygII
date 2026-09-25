"""Benchmark cache loading and seeded node splits for the six study datasets."""
import os.path as osp
import torch
from torch.utils.data import random_split
from lib_dataset import _single_datasets_
from lib_dataset.convert_datasets_to_pygDataset import dataset_Hypergraph


class HyperDataset:
    def __init__(self, args):
        self.device = args.device
        self.args = args
        self.name = args.dname
        self.method = args.method
        if self.name not in _single_datasets_:
            raise ValueError(f'Unsupported study dataset: {self.name}')
        self.load_data()

    def load_data(self):
        dname = self.name
        self.norm = None
        path = '../data/hete_data/' if dname == 'pokec' else '../data/trad_data'
        p2raw = osp.join(path, 'cocitation', '') if dname in ('cora', 'citeseer') else path
        noise = self.args.feature_noise if dname == 'house-committees-100' else None
        dataset = dataset_Hypergraph(name=dname,
                    root='../data/pyg_data/hypergraph_dataset_updated/',
                    p2raw=p2raw, feature_noise=noise)
        data = dataset.data
        if dname == 'house-committees-100':
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            data.num_hyperedges = torch.tensor([data.edge_index[0].max() - data.n_x + 1])
        self.data = data
        self.x = data.x
        self.hyperedge_index = data.edge_index
        self.y = data.y

    def _initialization_(self):
        
        self.num_classes= len(self.y.unique())
        self.num_features=self.x.shape[1]
        self.num_nodes=self.x.shape[0]
        self.num_hyperedges=len(self.hyperedge_index[1].unique())


    def to(self, device: str):
        
        self.x = self.x.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.y = self.y.to(device)
            
        return self


    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed = None):

        num_train = int(self.num_nodes * train_ratio)
        num_val = int(self.num_nodes * val_ratio)
        num_test = self.num_nodes - (num_train + num_val)

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.default_generator

        train_set, val_set, test_set = random_split(
            torch.arange(0, self.num_nodes), (num_train, num_val, num_test), 
            generator=generator)
        train_idx, val_idx, test_idx = \
            train_set.indices, val_set.indices, test_set.indices
        train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
        val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
        test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        '''
        [tensor([False, False, False,  ..., False, False, False], device='cuda:0'),
        tensor([False, False,  True,  ..., False,  True,  True], device='cuda:0'),
        tensor([ True,  True, False,  ...,  True, False, False], device='cuda:0')]
        '''
        return {'train':train_mask, 'valid': val_mask, 'test': test_mask}

