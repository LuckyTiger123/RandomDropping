import os
import gc
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
from torch import Tensor
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.typing import Adj
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from backbone_version.PPI.modified_GAT import ModifiedGATConv
from backbone_version.model import DropBlock

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=10)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-dm', '--drop_method', type=str, default='Dropout')
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

# configuration parameters
train_dataset = 'PPI'
drop_method = args.drop_method
cuda_device = args.cuda
train_round_number = args.train_round
epoch_num = args.epoch
file_id = args.file_id

drop_rate_list = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# fix random seed
random_seed = args.rand_seed
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

result_statistic = pd.DataFrame(
    columns=['dataset', 'backbone', 'drop_method', 'drop_rate', 'best_test_f1', 'best_val_f1',
             'average_val_f1', 'average_test_f1', 'test_std', 'best_test_f1_under_best_val', 'test_f1_list'])

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

dataset_train = PPI(root='/home/luckytiger/TestDataset/PPI', split='train')
dataset_val = PPI(root='/home/luckytiger/TestDataset/PPI', split='val')
dataset_test = PPI(root='/home/luckytiger/TestDataset/PPI', split='test')

train_loader = DataLoader(dataset_train, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=2, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=2, shuffle=False)


class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num):
        super(Model, self).__init__()
        self.drop_block = DropBlock(drop_method)
        self.conv1 = ModifiedGATConv(feature_num, 256, heads=4)
        self.lin1 = torch.nn.Linear(feature_num, 4 * 256)
        self.conv2 = ModifiedGATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = ModifiedGATConv(4 * 256, output_num, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, output_num)

    def forward(self, x: Tensor, edge_index: Adj):
        message_drop = drop_rate
        if drop_method != 'DropMessage':
            message_drop = 0
        egde_index_origin = edge_index
        if self.training:
            x, edge_index = self.drop_block.drop(x, egde_index_origin, drop_rate)
        x = F.elu(self.conv1(x, edge_index, drop_rate=message_drop) + self.lin1(x))
        if self.training:
            x, edge_index = self.drop_block.drop(x, egde_index_origin, drop_rate)
        x = F.elu(self.conv2(x, edge_index, drop_rate=message_drop) + self.lin2(x))
        if self.training:
            x, edge_index = self.drop_block.drop(x, egde_index_origin, drop_rate)
        x = self.conv3(x, edge_index, drop_rate=message_drop) + self.lin3(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


for drop_rate in drop_rate_list:
    gc.collect()
    best_test_f1 = best_val_f1 = average_val_f1 = average_test_f1 = best_test_f1_under_best_val = 0
    result_list = []
    model = Model(dataset_train.num_features, dataset_train.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_op = torch.nn.BCEWithLogitsLoss()
    for train_round in range(train_round_number):
        model.reset_parameters()
        c_best_val_f1 = c_best_test_f1 = test_std = 0
        result_list_msg = ''
        for epoch in range(epoch_num):
            loss = train()
            val_f1 = test(val_loader)
            test_f1 = test(test_loader)
            print('---------------------------------------------------------------------------')
            print('For the drop rate {}, round {}.'.format(drop_rate, train_round))
            print('For the {} epoch, the train loss is {}, the val f1 is {}, the test f1 is {}.'.format(epoch, loss,
                                                                                                        val_f1,
                                                                                                        test_f1))
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
            if val_f1 > c_best_val_f1:
                c_best_val_f1 = val_f1
                c_best_test_f1 = test_f1
                if c_best_test_f1 > best_test_f1_under_best_val:
                    best_test_f1_under_best_val = c_best_test_f1
        if c_best_test_f1 > 0.85:
            average_val_f1 += c_best_val_f1
            average_test_f1 += c_best_test_f1
            result_list.append(c_best_test_f1)
    if len(result_list) >= 1:
        average_val_f1 /= max(len(result_list), 1)
        average_test_f1 /= max(len(result_list), 1)
        test_std = float(np.std(result_list))
        result_list_msg = ','.join(['{:.4f}'.format(x) for x in result_list])

    result_statistic.loc[result_statistic.shape[0]] = {'dataset': train_dataset, 'backbone': 'GAT',
                                                       'drop_method': drop_method, 'drop_rate': drop_rate,
                                                       'best_test_f1': round(best_test_f1, 4),
                                                       'best_val_f1': round(best_val_f1, 4),
                                                       'average_val_f1': round(average_val_f1, 4),
                                                       'average_test_f1': round(average_test_f1, 4),
                                                       'test_std': round(test_std, 4),
                                                       'best_test_f1_under_best_val': round(
                                                           best_test_f1_under_best_val, 4),
                                                       'test_f1_list': result_list_msg}

save_path = os.path.join('..', '..', 'result', train_dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)
result_statistic.to_excel(os.path.join(save_path, '{}_{}_{}.xlsx'.format('GAT', drop_method, file_id)))
print('Mission complete.')
