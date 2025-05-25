import torch
import torch.nn.functional as F
import torch_sparse
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import scipy.sparse as sp


def rand_train_test_idx(label, train_ratio=0.5, val_ratio=0.25, ignore_negative=True):
    """randomly splits label into train/valid/test splits"""
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_ratio)
    valid_num = int(n * val_ratio)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num: train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    """use all remaining data points as test data, so test_num will not be used"""
    train_idx, non_train_idx = [], []
    idx = torch.arange(label.shape[0])
    class_list = label.squeeze().unique()
    for i in range(class_list.shape[0]):
        c_i = class_list[i]
        idx_i = idx[label.squeeze() == c_i]
        n_i = idx_i.shape[0]
        rand_idx = idx_i[torch.randperm(n_i)]
        train_idx += rand_idx[:label_num_per_class].tolist()
        non_train_idx += rand_idx[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    non_train_idx = non_train_idx[torch.randperm(non_train_idx.shape[0])]
    valid_idx, test_idx = (
        non_train_idx[:valid_num],
        non_train_idx[valid_num: valid_num + test_num],)
    print(f"train:{train_idx.shape}, valid:{valid_idx.shape}, test:{test_idx.shape}")
    return train_idx, valid_idx, test_idx


def load_fixed_splits(data, data_dir, name):
    splits = {}
    splits_lst = []
    if name in ["film", "cornell", "texas", "wisconsin", 'chameleon', 'squirrel', 'roman-empire']:
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:, i])[0]
            splits['valid'] = torch.where(data.val_mask[:, i])[0]
            splits['test'] = torch.where(data.test_mask[:, i])[0]
            splits_lst.append(splits)

    elif name in ['ogbn-arxiv', 'ogbn-products']:
        split_idx = data.get_idx_split()
        splits['train'] = torch.as_tensor(split_idx['train'])
        splits['valid'] = torch.as_tensor(split_idx['valid'])
        splits['test'] = torch.as_tensor(split_idx['test'])
    else:
        raise NotImplementedError
    return splits['train'], splits['valid'], splits['test'], splits_lst


def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)
    return sum(acc_list) / len(acc_list)


def eval_acc(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')
    return sum(rocauc_list) / len(rocauc_list)

def partition(edge_index, nodes, clusters):
    adj = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(nodes, nodes))
    _, partptr, perm = adj.partition(clusters, recursive=False)
    partptr = partptr.tolist()
    perm = perm.tolist()
    partptr = list(set(partptr))
    partptr.sort()
    return partptr, perm, len(partptr) - 1


def partition_arxiv(adj, clusters):
    adj = adj.coalesce()
    row, col = adj.indices()
    values = adj.values()
    size = adj.size()
    adj = torch_sparse.SparseTensor(row=row, col=col, value=values, sparse_sizes=size)
    _, partptr, perm = adj.partition(clusters, recursive=False)

    partptr = partptr.tolist()
    perm = perm.tolist()
    partptr = list(set(partptr))
    partptr.sort()
    return partptr, perm, len(partptr) - 1
def partition_adj(adj, clusters):
    adj = adj.coalesce()
    row, col = adj.indices()
    values = adj.values()
    size = adj.size()
    adj = torch_sparse.SparseTensor(row=row, col=col, value=values, sparse_sizes=size)
    _, partptr, perm = adj.partition(clusters, recursive=False)

    partptr = partptr.tolist()
    perm = perm.tolist()
    partptr = list(set(partptr))
    partptr.sort()
    return partptr, perm, len(partptr) - 1

def adj_par(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    adj = sp.csr_matrix(([1]*edge_index.size(1), (edge_index[0].numpy(), edge_index[1].numpy())),
                        shape=(num_nodes, num_nodes), dtype=np.float32)

    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def sym_adj(edge_index, num_nodes):
    edge_index = to_undirected(edge_index)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = sp.csr_matrix(([1]*edge_index.size(1), (edge_index[0].numpy(), edge_index[1].numpy())),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #data.x = normalize(data.x)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


dataset_drive_url = {
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}
