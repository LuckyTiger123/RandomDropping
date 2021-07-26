import math
import random
import os.path as osp
from itertools import chain

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
from torch.nn import BCEWithLogitsLoss
from torch.nn import Linear, MaxPool1d

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import global_sort_pool
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   train_test_split_edges, k_hop_subgraph,
                                   to_scipy_sparse_matrix)
import torch_geometric.transforms as T

import backbone_version.model as Md
import backbone_version.seal_dataset as SD


# Model
class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.backbone = backbone
        self.gnn1 = Md.OurModelLayer(feature_num, 64, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)
        self.gnn2 = Md.OurModelLayer(64, output_num, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)
        self.maxpool1d = MaxPool1d(2, 2)
        self.linear = Linear(128, 1)

    def forward(self, x: Tensor, edge_index: Adj, batch, drop_rate: float = 0):
        x = self.gnn1(x, edge_index, drop_rate)
        x = self.gnn2(x, edge_index, drop_rate)
        x = global_sort_pool(x, batch, k=64)
        x = self.linear(x)
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


print("Params init...")
train_dataset = 'Cora' # ['Cora', 'CiteSeer', 'PubMed']
backbone = 'GCN'  # ['GAT', 'GCN', 'APPNP']:
drop_method = 'DropMessage' # ['DropNode', 'DropEdge', 'Dropout', 'DropMessage']:
drop_rate = 0  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
unbias = True

print("Dataset init...")
dataset = Planetoid(root="../TestDataset/{}".format(train_dataset),
                    name=train_dataset,
                    transform=T.NormalizeFeatures())
# device selection
cuda_device = 0
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

epoch_num = 50

print("SEALDataset process...")
# process dataset for link prediction
train_dataset = SD.SEALDataset(dataset, num_hops=2, split='train')
val_dataset = SD.SEALDataset(dataset, num_hops=2, split='val')
test_dataset = SD.SEALDataset(dataset, num_hops=2, split='test')

print("Model init...")
unbias_rate = drop_rate if unbias else 0
model = Model(train_dataset.num_features, train_dataset.num_classes, backbone, drop_method, unbias_rate).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=0.0005)

print("DataLoader init...")
# load data for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

print("Train begin...")
def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index,  data.batch)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred))


best_val_auc = test_auc = 0
for epoch in range(epoch_num):
    loss = train()
    val_auc = test(val_loader)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        test_auc = test(test_loader)

    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print('Mission completes.')
print('The best val acc is {}, and the test acc is {}.'.format(best_val_auc, test_auc))

