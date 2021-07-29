import os
import gc
import sys
import torch
import random
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

cuda_device = 4

train_dataset = 'elliptic'
drop_method = 'DropMessage'
drop_rate = 0.05
backbone = 'GCN'
unbias = False

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# random generate train, validate, test mask
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

dataset_path = '/data/luckytiger/elliptic_bitcoin_dataset'
# Data Pretreatment
df_features = pd.read_csv(dataset_path + '/elliptic_txs_features.csv', header=None)
df_edges = pd.read_csv(dataset_path + '/elliptic_txs_edgelist.csv')
df_classes = pd.read_csv(dataset_path + '/elliptic_txs_classes.csv')
df_classes['class'] = df_classes['class'].map({'unknown': 2, '1': 1, '2': 0})

# merging dataframes
df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
df_merge = df_merge.sort_values(0).reset_index(drop=True)
classified = df_merge.loc[df_merge['class'].loc[df_merge['class'] != 2].index].drop('txId', axis=1)
unclassified = df_merge.loc[df_merge['class'].loc[df_merge['class'] == 2].index].drop('txId', axis=1)

# storing classified unclassified nodes seperatly for training and testing purpose
classified_edges = df_edges.loc[df_edges['txId1'].isin(classified[0]) & df_edges['txId2'].isin(classified[0])]
unclassifed_edges = df_edges.loc[df_edges['txId1'].isin(unclassified[0]) | df_edges['txId2'].isin(unclassified[0])]
del df_features, df_classes

# all nodes in data
nodes = df_merge[0].values
map_id = {j: i for i, j in enumerate(nodes)}  # mapping nodes to indexes

edges = df_edges.copy()
edges.txId1 = edges.txId1.map(map_id)
edges.txId2 = edges.txId2.map(map_id)
edges = edges.astype(int)

edge_index = np.array(edges.values).T
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

# maping txIds to corresponding indexes, to pass node features to the model
node_features = df_merge.drop(['txId'], axis=1).copy()
node_features[0] = node_features[0].map(map_id)
classified_idx = node_features['class'].loc[node_features['class'] != 2].index
classified_0_idx = node_features['class'].loc[node_features['class'] == 0].index
classified_1_idx = node_features['class'].loc[node_features['class'] == 1].index
unclassified_idx = node_features['class'].loc[node_features['class'] == 2].index
# replace unkown class with 0, to avoid having 3 classes, this data/labels never used in training
node_features['class'] = node_features['class'].replace(2, 0)

labels = node_features['class'].values
node_features = torch.tensor(np.array(node_features.drop([0, 'class', 1], axis=1).values, dtype=np.double),
                             dtype=torch.double)

# converting data to PyGeometric graph data format
data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                  y=torch.tensor(labels, dtype=torch.double))  # , adj= torch.from_numpy(np.array(adj))

data_train_mask = np.load(os.path.join(dataset_path, 'train_mask.npy')).tolist()
data_val_mask = np.load(os.path.join(dataset_path, 'val_mask.npy')).tolist()
data_test_mask = np.load(os.path.join(dataset_path, 'test_mask.npy')).tolist()


# Model
class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.backbone = backbone
        self.gnn1 = Md.OurModelLayer(feature_num, 128, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)
        self.gnn2 = Md.OurModelLayer(128, output_num, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        x = self.gnn1(x, edge_index, drop_rate)
        if self.backbone == 'GAT':
            x = F.elu(x)
        else:
            x = F.relu(x)
        x = self.gnn2(x, edge_index, drop_rate)
        return torch.sigmoid(x)

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


data = data_train.to(device)
data.x = data.x.float()
data.y = data.y.long()

if unbias:
    unbias_rate = drop_rate
else:
    unbias_rate = 0

model = Model(data_train.x.size(1), 1, backbone, drop_method, unbias_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
epoch_num = 500
criterion = torch.nn.BCELoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, drop_rate)
    out = out.reshape((data.x.shape[0]))
    loss = criterion(out[data_train_mask], data.y[data_train_mask].float())
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, drop_rate)
    train_auc = roc_auc_score(data.y.detach().cpu().numpy()[data_train_mask],
                              out.detach().cpu().numpy()[data_train_mask])
    val_auc = roc_auc_score(data.y.detach().cpu().numpy()[data_val_mask],
                            out.detach().cpu().numpy()[data_val_mask])
    test_auc = roc_auc_score(data.y.detach().cpu().numpy()[data_test_mask],
                             out.detach().cpu().numpy()[data_test_mask])

    return train_auc, val_auc, test_auc


# def train():
#     model.train()
#     optimizer.zero_grad()
#     out = model(data.x, data.edge_index, drop_rate)
#     loss = F.cross_entropy(out[data_train_mask], data.y[data_train_mask], weight=Tensor([1, 10]).to(device))
#     loss.backward()
#     print('the train loss is {}'.format(float(loss)))
#     optimizer.step()
#
#
# @torch.no_grad()
# def test():
#     model.eval()
#     out = model(data.x, data.edge_index, drop_rate)
#     _, pred = out.max(dim=1)
#     train_correct = int(pred[data_train_mask].eq(data.y[data_train_mask]).sum().item())
#     train_acc = train_correct / int(len(data_train_mask))
#     validate_correct = int(pred[data_val_mask].eq(data.y[data_val_mask]).sum().item())
#     validate_acc = validate_correct / int(len(data_val_mask))
#     test_correct = int(pred[data_test_mask].eq(data.y[data_test_mask]).sum().item())
#     test_acc = test_correct / int(len(data_test_mask))
#     # train_1_num = data.y[data_train_mask][data.y[data_train_mask] == 1].size(0)
#     # train_pred_1_num = pred[data_train_mask][pred[data_train_mask] == 1].size(0)
#     # train_correct_1_num = pred[data_train_mask][data.y[data_train_mask] == 1].eq(1).sum().item()
#     # train_f1 = 2 * train_correct_1_num * train_correct_1_num / ((train_correct_1_num * train_1_num) + (
#     #         train_correct_1_num * train_pred_1_num))
#     # val_1_num = data.y[data_val_mask][data.y[data_val_mask] == 1].size(0)
#     # val_pred_1_num = pred[data_val_mask][pred[data_val_mask] == 1].size(0)
#     # val_correct_1_num = pred[data_val_mask][data.y[data_val_mask] == 1].eq(1).sum().item()
#     # val_f1 = 2 * val_correct_1_num * val_correct_1_num / ((val_correct_1_num * val_1_num) + (
#     #         val_correct_1_num * val_pred_1_num))
#     # test_1_num = data.y[data_test_mask][data.y[data_test_mask] == 1].size(0)
#     # test_pred_1_num = pred[data_test_mask][pred[data_test_mask] == 1].size(0)
#     # test_correct_1_num = pred[data_test_mask][data.y[data_test_mask] == 1].eq(1).sum().item()
#     # test_f1 = 2 * test_correct_1_num * test_correct_1_num / ((test_correct_1_num * test_1_num) + (
#     #         test_correct_1_num * test_pred_1_num))
#
#     return train_acc, validate_acc, test_acc


best_val_acc = test_acc = 0
for epoch in range(epoch_num):
    train()
    train_acc, val_acc, current_test_acc = test()
    print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch, train_acc,
                                                                                                 val_acc,
                                                                                                 current_test_acc))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = current_test_acc

print('Mission completes.')
print('The best val acc is {}, and the test acc is {}.'.format(best_val_acc, test_acc))
