import numpy as np
import torch
import csv
import torch.utils.data.dataset as Dataset
import os

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def Simdata_processing(param):
    dataset = dict()

    mm_funsim = read_csv(param.datapath + '/m_fs.csv')
    dataset['mm_f'] = mm_funsim

    mm_seqsim = read_csv(param.datapath + '/m_ss.csv')
    dataset['mm_s'] = mm_seqsim

    mm_gausim = read_csv(param.datapath + '/m_gs.csv')
    dataset['mm_g'] = mm_gausim

    dd_funsim = read_csv(param.datapath + '/d_ts.csv')
    dataset['dd_t'] = dd_funsim

    dd_semsim = read_csv(param.datapath + '/d_ss.csv')
    dataset['dd_s'] = dd_semsim

    dd_gausim = read_csv(param.datapath + '/d_gs.csv')
    dataset['dd_g'] = dd_gausim

    return dataset


def Simdata_pro(param):
    dataset = dict()
    mm_funsim = np.loadtxt(param.datapath + 'm_fs.csv', dtype=float, delimiter=',')
    mm_seqsim = np.loadtxt(param.datapath + 'm_ss.csv', dtype=float, delimiter=',')
    mm_gausim = np.loadtxt(param.datapath + 'm_gs.csv', dtype=float, delimiter=',')
    dd_funsim = np.loadtxt(param.datapath + 'd_ts.csv', dtype=float, delimiter=',')
    dd_semsim = np.loadtxt(param.datapath + 'd_ss.csv', dtype=float, delimiter=',')
    dd_gausim = np.loadtxt(param.datapath + 'd_gs.csv', dtype=float, delimiter=',')

    dataset['mm_f'] = torch.FloatTensor(mm_funsim)
    dataset['mm_s'] = torch.FloatTensor(mm_seqsim)
    dataset['dd_t'] = torch.FloatTensor(dd_funsim)
    dataset['dd_s'] = torch.FloatTensor(dd_semsim)

    dataset['mm_g'] = torch.FloatTensor(mm_gausim)
    dataset['dd_g'] = torch.FloatTensor(dd_gausim)

    return dataset


def load_data(param):
    # Load the original miRNA-disease associations matrix.
    md_matrix = np.loadtxt(param.datapath + '/m_d.csv', dtype=np.float32, delimiter=',')

    rng = np.random.default_rng(seed=42)
    pos_samples = np.where(md_matrix == 1)

    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    rng = np.random.default_rng(seed=42)
    neg_samples = np.where(md_matrix == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]
    idx_split = int(n_pos_samples * param.ratio)

    test_pos_edges = pos_samples_shuffled[:, :idx_split]
    test_neg_edges = neg_samples_shuffled[:, :idx_split]
    test_pos_edges = test_pos_edges.T
    test_neg_edges = test_neg_edges.T
    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))
    # np.savetxt('./train_test/test_pos.csv', test_pos_edges, delimiter=',')
    # np.savetxt('./train_test/test_neg.csv', test_neg_edges, delimiter=',')

    train_pos_edges = pos_samples_shuffled[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))
    # np.savetxt('./train_test/train_pos.csv', train_pos_edges, delimiter=',')
    # np.savetxt('./train_test/train_neg.csv', train_neg_edges, delimiter=',')

    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label

    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label

    # Load the collected miRNA-disease associations matrix with edge attributes.
    md_class = np.loadtxt(param.datapath + '/m_d_edge.csv', dtype=np.float32, delimiter=',')
    edge_idx_dict['md_class'] = md_class
    edge_idx_dict['true_md'] = md_matrix

    return edge_idx_dict


class EdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):
        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
