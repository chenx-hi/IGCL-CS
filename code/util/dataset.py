import os

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset, WikiCS, WebKB, WikipediaNetwork, Actor
from ogb.nodeproppred import NodePropPredDataset
from os import path
from torch_geometric.datasets import Planetoid
from .data_utils import class_rand_splits, load_fixed_splits, rand_train_test_idx, sym_adj, partition, adj_par, \
    partition_adj, partition_arxiv

class Dataset(object):
    def __init__(self, name, graph, x, y, idx_train, idx_valid, idx_test, splits_lst, partptr, perm, cluster):
        self.name = name  # original name, e.g., cora
        self.graph = graph
        self.x = x
        self.y = y
        self.num_nodes = x.size(0)
        self.num_features = x.size(1)
        self.num_classes = y.max() + 1
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test = idx_test
        self.splits_lst = splits_lst
        self.partptr = partptr
        self.perm = perm
        self.cluster = cluster
        self.partition = None
        self.par_lable = None
        self.coarse_g = None

    def to_train(self, device):
        #self.graph = self.graph.to(device)
        self.x = self.x.to(device)
        #self.y = self.y.to(device)

        node_to_par = torch.zeros(self.num_nodes, dtype=torch.long)
        par_value = torch.ones(self.num_nodes)
        for i in range(self.cluster):
            start_idx = self.partptr[i]
            end_idx = self.partptr[i + 1]
            nodes_size = end_idx - start_idx
            node_to_par[self.perm[start_idx:end_idx]] = i
            par_value[self.perm[start_idx:end_idx]] = 1 / nodes_size
        par_indices = torch.stack([torch.arange(self.num_nodes), node_to_par])
        partition = torch.sparse.FloatTensor(par_indices, par_value,
                                             (self.num_nodes, self.cluster))
        self.partition = partition
        P_A = torch.spmm(partition.T, self.graph)
        self.coarse_g = torch.spmm(P_A, partition).to(device)
        self.partition = partition.to(device)
        return self

    def to_test(self, device):
        self.graph = self.graph.to(device)
        self.y = self.y.to(device)
        return self

class Ogbn_Dataset(object):
    def __init__(self, name, graph, x, y, idx_train, idx_valid, idx_test, par):
        self.name = name  # original name, e.g., cora
        self.graph = graph
        self.x = x
        self.y = y
        self.num_nodes = x.size(0)
        self.num_features = x.size(1)
        self.num_classes = y.max() + 1
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test = idx_test
        self.partition = par
        self.cluster = par.size(1)
        self.coarse_g = None

    def to_train(self, device):
        #self.graph = self.graph.to(device)
        self.x = self.x.to(device)
        #self.y = self.y.to(device)
        P_A = torch.spmm(self.partition.T, self.graph)
        self.coarse_g = torch.spmm(P_A, self.partition).to(device)
        self.partition = self.partition.to(device)
        return self

    def to_test(self, device):
        self.graph = self.graph.to(device)
        self.y = self.y.to(device)
        return self

def load_data(args):
    data_name = args.dataset_name
    print("Loading {} dataset...".format(data_name))
    data_dir = '../data/'

    if data_name in  ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, data_name)
    elif data_name in  ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, data_name)
    elif data_name in ('roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'):
        dataset = load_hetero_dataset(data_dir, data_name)
    elif data_name == 'wikics':
        dataset = WikiCS(root=f'{data_dir}/wikics/')
    elif data_name in ('cora', 'citeseer', 'pubmed'):
        dataset = Planetoid(root=f'{data_dir}/Planetoid', name=data_name, transform=T.NormalizeFeatures())
    elif data_name in ["cornell", "texas", "wisconsin"]:
        dataset = WebKB(root=f'{data_dir}/webkb/', name=data_name)
    elif data_name in ['squirrel', 'chameleon']:
        dataset = WikipediaNetwork(root=f'{data_dir}/datasets_new/', name=data_name, geom_gcn_preprocess=True)
    elif data_name == 'film':
        dataset = Actor(root=f'{data_dir}/film/', transform=T.NormalizeFeatures())
    else:
        raise ValueError('Invalid dataname')
    data = dataset[0]

    # print(our_measure(data.edge_index, data.y))
    # exit(0)

    adj = sym_adj(data.edge_index, data.num_nodes)
    #data.x = torch.nn.functional.normalize(data.x)
    if data_name in ["texas", "wisconsin", 'chameleon', 'film', 'roman-empire']:
        adj_partition = adj_par(data.edge_index, data.num_nodes)
        partptr, perm, cluster = partition_adj(adj_partition, round(args.cluster * data.x.size(0)))
    else:
        partptr, perm, cluster = partition(data.edge_index, data.num_nodes, round(args.cluster*data.x.size(0)))


    #data.x = torch.nn.functional.normalize(data.x)
    splits_lst = []
    if args.rand_split_class:
        idx_train, idx_valid, idx_test = \
            class_rand_splits(data.y, args.label_num_per_class, args.valid_num, args.test_num)
    elif args.rand_split:
        idx_train, idx_valid, idx_test = rand_train_test_idx(data.y, args.train_ratio, args.valid_ratio)
    else:
        idx_train, idx_valid, idx_test, splits_lst = load_fixed_splits(data, data_dir, name=data_name)

    # idx_train = torch.where(data.train_mask)[0]
    # idx_valid = torch.where(data.val_mask)[0]
    # idx_test = torch.where(data.test_mask)[0]

    return Dataset(data_name, adj, data.x, data.y, idx_train, idx_valid, idx_test, splits_lst, partptr, perm, cluster)


def load_ogbn_data(args):
    data_name = args.dataset_name
    print("Loading {} dataset...".format(data_name))
    data_dir = '../data/'

    if data_name in ['ogbn-arxiv', 'ogbn-products']:
        dataset = NodePropPredDataset(root=f'{data_dir}/ogb/',name=data_name)
    else:
        raise ValueError('Invalid dataname')

    data, y = dataset[0]
    n = data['num_nodes']
    data['edge_index'] = torch.as_tensor(data['edge_index'])
    adj = sym_adj(torch.as_tensor(data['edge_index']), n)
    y = torch.squeeze(torch.as_tensor(y).reshape(-1, 1), dim=1)
    x = torch.as_tensor(data['node_feat'])

    par_dir = f'{data_dir}/ogb/' + data_name + '/' + str(args.cluster)
    if (os.path.exists( par_dir +"/partition.pt")):
        par = torch.load(par_dir +"/partition.pt")
        #print('aaa')
    else:
        if data_name == 'ogbn-arxiv':
            adj_partition = adj_par(torch.as_tensor(data['edge_index']), n)

            partptr, perm, cluster = partition_arxiv(adj_partition, round(args.cluster * x.size(0)))
        else:
            partptr, perm, cluster = partition(data['edge_index'], n, round(args.cluster * x.size(0)))
        #partptr, perm, cluster = partition(data['edge_index'], n, round(args.cluster * x.size(0)))
        node_to_par = torch.zeros(n, dtype=torch.long)
        par_value = torch.ones(n)
        for i in range(cluster):
            start_idx = partptr[i]
            end_idx = partptr[i + 1]
            nodes_size = end_idx - start_idx
            node_to_par[perm[start_idx:end_idx]] = i
            par_value[perm[start_idx:end_idx]] = 1 / nodes_size
        par_indices = torch.stack([torch.arange(n), node_to_par])
        par = torch.sparse.FloatTensor(par_indices, par_value, (n, cluster))
        par_path = os.path.join(par_dir, 'partition.pt')
        os.makedirs(os.path.dirname(par_path), exist_ok=True)
        torch.save(par, par_dir +"/partition.pt")

    if args.rand_split_class:
        idx_train, idx_valid, idx_test = \
            class_rand_splits(data.y, args.label_num_per_class, args.valid_num, args.test_num)
    elif args.rand_split:
        idx_train, idx_valid, idx_test = rand_train_test_idx(data.y, args.train_ratio, args.valid_ratio)
    else:
        idx_train, idx_valid, idx_test, _ = load_fixed_splits(dataset, data_dir, name=data_name)

    return Ogbn_Dataset(data_name, adj, x, y, idx_train, idx_valid, idx_test, par)


def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        dataset = Amazon(root=f'{data_dir}Amazon', name='Photo', transform=transform)
        return dataset
    elif name == 'amazon-computer':
        dataset = Amazon(root=f'{data_dir}Amazon', name='Computers', transform=transform)
        return dataset


def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        dataset = Coauthor(root=f'{data_dir}Coauthor', name='CS', transform=transform)
        return dataset
    elif name == 'coauthor-physics':
        dataset = Coauthor(root=f'{data_dir}Coauthor', name='Physics', transform=transform)
        return dataset
