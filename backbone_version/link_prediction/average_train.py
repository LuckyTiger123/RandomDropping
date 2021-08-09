import os
import gc
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=10)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=500)
parser.add_argument('-d', '--dataset', type=str, default='Cora')
parser.add_argument('-bb', '--backbone', type=str, default='GCN')
parser.add_argument('-dm', '--drop_method', type=str, default='Dropout')
parser.add_argument('-hs', '--heads', type=int, default=1)
parser.add_argument('-k', '--K', type=int, default=10)
parser.add_argument('-a', '--alpha', type=float, default=0.1)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

# configuration parameters
train_dataset = args.dataset
backbone = args.backbone
drop_method = args.drop_method
cuda_device = args.cuda
train_round_number = args.train_round
epoch_num = args.epoch
heads = args.heads
K = args.K
alpha = args.alpha
file_id = args.file_id

drop_rate_list = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# fix random seed
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

result_statistic = pd.DataFrame(
    columns=['dataset', 'backbone', 'drop_method', 'drop_rate', 'average_val_auc', 'average_test_auc',
             'test_auc_std', 'best_test_auc_under_best_val'])

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/{}".format(train_dataset), name=train_dataset,
                    transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
data = data.to(device)


class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.backbone = backbone
        self.gnn1 = Md.OurModelLayer(feature_num, 128, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)
        self.gnn2 = Md.OurModelLayer(128, output_num, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)

    def encode(self, x, edge_index, drop_rate: float = 0):
        x = self.gnn1(x, edge_index, drop_rate)
        x = x.relu()
        return self.gnn2(x, edge_index, drop_rate)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(data):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index, drop_rate)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(data):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index, drop_rate)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results


for drop_rate in drop_rate_list:
    gc.collect()
    average_val_auc = average_test_auc = best_test_auc_under_best_val = 0
    if drop_method == 'DropNode':
        unbias_rate = drop_rate
    else:
        unbias_rate = 0

    result_auc_list = []
    model = Model(dataset.num_features, 64, backbone, drop_method, unbias_rate).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    for train_round in range(train_round_number):
        model.reset_parameters()
        gc.collect()
        best_val_auc = best_test_auc = 0
        for epoch in range(epoch_num):
            loss = train(data)
            val_auc, tmp_test_auc = test(data)
            print('---------------------------------------------------------------------------')
            print('For the drop rate {},  round {}, epoch {}.'.format(drop_rate, train_round, epoch))
            print('The train loss is {}.'.format(float(loss)))
            print('The val auc is {}, the test auc is {}.'.format(val_auc, tmp_test_auc))
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_test_auc = tmp_test_auc
        average_val_auc += best_val_auc
        average_test_auc += best_test_auc
        result_auc_list.append(best_test_auc)
        if best_test_auc > best_test_auc_under_best_val:
            best_test_auc_under_best_val = best_test_auc
    average_val_auc /= train_round_number
    average_test_auc /= train_round_number
    test_auc_std = float(np.std(result_auc_list))
    result_statistic.loc[result_statistic.shape[0]] = {'dataset': train_dataset, 'backbone': backbone,
                                                       'drop_method': drop_method, 'drop_rate': drop_rate,
                                                       'average_val_auc': round(average_val_auc, 4),
                                                       'average_test_auc': round(average_test_auc, 4),
                                                       'test_auc_std': round(test_auc_std, 4),
                                                       'best_test_auc_under_best_val': round(
                                                           best_test_auc_under_best_val, 4)}

save_path = os.path.join('..', '..', 'result', 'link_pred', train_dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)
result_statistic.to_excel(os.path.join(save_path, '{}_{}_{}.xlsx'.format(backbone, drop_method, file_id)))
print('Mission complete.')
