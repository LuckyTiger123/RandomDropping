import os
import sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
from torch.nn import BCEWithLogitsLoss
from torch.nn import Linear, MaxPool1d

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import global_sort_pool
from torch_geometric.data import DataLoader
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


def train(loader, dataset):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(dataset)


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


print("Params init...")
train_dataset = 'CiteSeer' # ['Cora', 'CiteSeer', 'PubMed']
#backbone = 'APPNP'  # ['GAT', 'GCN', 'APPNP']:
drop_method = 'DropEdge' # ['DropNode', 'DropEdge', 'Dropout', 'DropMessage']:
# drop_rate = 0  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
unbias = True

# device selection
cuda_device = 1
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

epoch_num = 50
train_round_num = 10

result_statistics = pd.DataFrame(
    columns=['dataset', 'backbone', 'drop_method', 'drop_rate', 'unbias',
             'train_round', 'best_val_auc', 'test_auc']
)

for backbone in ['GAT', 'GCN', 'APPNP']:
#for drop_method in ['DropMessage', 'DropNode', 'Dropout', 'DropEdge']:

    print("Dataset init: {}, {}, {}...".format(train_dataset, backbone, drop_method))
    dataset = Planetoid(root="../TestDataset/{}".format(train_dataset),
                        name=train_dataset,
                        transform=T.NormalizeFeatures())

    # process dataset for link prediction
    print("SEALDataset process...")
    train_dataset_ = SD.SEALDataset(dataset, num_hops=2, split='train')
    val_dataset_ = SD.SEALDataset(dataset, num_hops=2, split='val')
    test_dataset_ = SD.SEALDataset(dataset, num_hops=2, split='test')

    for drop_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("Model init: drop_rate {}...".format(drop_rate))
        unbias_rate = drop_rate if unbias else 0
        model = Model(train_dataset_.num_features,
                      train_dataset_.num_classes,
                      backbone,
                      drop_method,
                      unbias_rate).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005, weight_decay=0.0005)

        # load data for training
        print("DataLoader init...")
        train_loader = DataLoader(train_dataset_, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset_, batch_size=32)
        test_loader = DataLoader(test_dataset_, batch_size=32)

        for train_round in range(train_round_num):
            print("Train begin: train_round {}...".format(train_round))
            best_val_auc = test_auc = 0
            for epoch in range(epoch_num):
                loss = train(train_loader, train_dataset_)
                val_auc = test(val_loader)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    test_auc = test(test_loader)

                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
                      f'Test: {test_auc:.4f}')

            result_statistics.loc[result_statistics.shape[0]] = {
                'dataset': train_dataset,
                'backbone': backbone,
                'drop_method': drop_method,
                'drop_rate': drop_rate,
                'unbias': unbias,
                'best_val_auc': round(best_val_auc, 4),
                'test_auc': round(test_auc, 4),
                'train_round': train_round
            }
            print('The best val acc is {}, and the test acc is {}.'.format(best_val_auc, test_auc))

    save_path = os.path.join('..', 'result_linkpred_seal_0802', train_dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_statistics.to_excel(os.path.join(save_path, '{}_{}_linkpred_{}.xlsx'.format(backbone, drop_method, unbias)))
    print('File saved!')

print('Mission completes.')
