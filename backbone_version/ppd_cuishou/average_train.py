import os
import gc
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import torch_geometric.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=10)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-e', '--epoch', type=int, default=500)
parser.add_argument('-bb', '--backbone', type=str, default='GCN')
parser.add_argument('-dm', '--drop_method', type=str, default='Dropout')
parser.add_argument('-hd', '--hidden_dimensions', type=int, default=64)
parser.add_argument('-nl', '--num_layers', type=int, default=3)
parser.add_argument('-hs', '--heads', type=int, default=1)
parser.add_argument('-k', '--K', type=int, default=5)
parser.add_argument('-a', '--alpha', type=float, default=0.1)
parser.add_argument('-m', '--month', type=str, default='05')
parser.add_argument('-dd', '--date_day', type=str, default='01')
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

direct_path = '/data/luckytiger/ppd_cuishou/graph/{}_{}'.format(args.month, args.date_day)

# configuration parameters
train_dataset = 'ppd_cuishou'
backbone = args.backbone
drop_method = args.drop_method
cuda_device = args.cuda
train_round_number = args.train_round
epoch_num = args.epoch
heads = args.heads
K = args.K
alpha = args.alpha
file_id = args.file_id
hidden_dimensions = args.hidden_dimensions
num_layers = args.num_layers

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# fix random seed
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# data collect
node_x = torch.from_numpy(np.load(os.path.join(direct_path, 'node_x.npy'))).float().to(device)
node_y = torch.from_numpy(np.load(os.path.join(direct_path, 'node_y.npy'))).t().to(device)
graph_edge_index = torch.from_numpy(np.load(os.path.join(direct_path, 'edge_index.npy'))).to(device)
train_mask = torch.from_numpy(np.load(os.path.join(direct_path, 'train_mask.npy'))).to(device)
val_mask = torch.from_numpy(np.load(os.path.join(direct_path, 'val_mask.npy'))).to(device)
test_mask = torch.from_numpy(np.load(os.path.join(direct_path, 'test_mask.npy'))).to(device)
label_train_mask = torch.from_numpy(np.load(os.path.join(direct_path, 'label_train_mask.npy'))).to(device)
label_val_mask = torch.from_numpy(np.load(os.path.join(direct_path, 'label_val_mask.npy'))).to(device)
label_test_mask = torch.from_numpy(np.load(os.path.join(direct_path, 'label_test_mask.npy'))).to(device)

# test list
drop_rate_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
label_select_list = [0, 1]

# data table
result_statistic = pd.DataFrame(
    columns=['dataset', 'date', 'label', 'backbone', 'drop_method', 'drop_rate', 'average_val_acc', 'average_test_acc',
             'test_acc_std', 'best_test_acc_under_best_val', 'average_val_f1', 'average_test_f1', 'test_f1_std',
             'best_test_f1_under_best_val'])


# Model
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            Md.OurModelLayer(in_channels, hidden_channels, drop_method, backbone, unbias=unbias))
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(Md.OurModelLayer(hidden_channels, hidden_channels, drop_method, backbone, unbias=unbias))
            self.batch_norms.append(nn.BatchNorm(hidden_channels))
        self.convs.append(Md.OurModelLayer(hidden_channels, out_channels, drop_method, backbone, unbias=unbias))
        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, drop_rate)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(node_x, graph_edge_index, drop_rate)
    loss = F.cross_entropy(out[train_mask], node_y[label_select][label_train_mask], weight=Tensor([1, 2]).to(device))
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(node_x, graph_edge_index, drop_rate)
    _, pred = out.max(dim=1)
    # acc
    train_correct = int(pred[train_mask].eq(node_y[label_select][label_train_mask]).sum().item())
    train_acc = train_correct / len(train_mask)
    validate_correct = int(pred[val_mask].eq(node_y[label_select][label_val_mask]).sum().item())
    validate_acc = validate_correct / len(val_mask)
    test_correct = int(pred[test_mask].eq(node_y[label_select][label_test_mask]).sum().item())
    test_acc = test_correct / len(test_mask)

    # f1
    train_1_num = node_y[label_select][label_train_mask][node_y[label_select][label_train_mask] == 1].size(0)
    train_pred_1_num = pred[train_mask][pred[train_mask] == 1].size(0)
    train_correct_1_num = pred[train_mask][node_y[label_select][label_train_mask] == 1].eq(1).sum().item()
    train_f1 = 2 * train_correct_1_num * train_correct_1_num / (0.0001 + (train_correct_1_num * train_1_num) + (
            train_correct_1_num * train_pred_1_num))

    val_1_num = node_y[label_select][label_val_mask][node_y[label_select][label_val_mask] == 1].size(0)
    val_pred_1_num = pred[val_mask][pred[val_mask] == 1].size(0)
    val_correct_1_num = pred[val_mask][node_y[label_select][label_val_mask] == 1].eq(1).sum().item()
    val_f1 = 2 * val_correct_1_num * val_correct_1_num / (0.0001 + (val_correct_1_num * val_1_num) + (
            val_correct_1_num * val_pred_1_num))

    test_1_num = node_y[label_select][label_test_mask][node_y[label_select][label_test_mask] == 1].size(0)
    test_pred_1_num = pred[test_mask][pred[test_mask] == 1].size(0)
    test_correct_1_num = pred[test_mask][node_y[label_select][label_test_mask] == 1].eq(1).sum().item()
    test_f1 = 2 * test_correct_1_num * test_correct_1_num / (0.0001 + (test_correct_1_num * test_1_num) + (
            test_correct_1_num * test_pred_1_num))

    return train_acc, validate_acc, test_acc, train_f1, val_f1, test_f1


for label_select in label_select_list:
    for drop_rate in drop_rate_list:
        gc.collect()
        average_val_acc = average_test_acc = best_test_acc_under_best_val = average_val_f1 = average_test_f1 = best_test_f1_under_best_val = 0
        if drop_method == 'DropNode':
            unbias_rate = drop_rate
        else:
            unbias_rate = 0
        result_acc_list = []
        result_f1_list = []
        model = Model(node_x.size(1), hidden_dimensions, 2, num_layers, backbone, drop_method, unbias_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
        for train_round in range(train_round_number):
            model.reset_parameters()
            gc.collect()
            best_val_acc = best_test_acc = 0
            best_val_f1 = best_test_f1 = 0
            for epoch in range(epoch_num):
                train()
                train_acc, val_acc, test_acc, train_f1, val_f1, test_f1 = test()
                print('---------------------------------------------------------------------------')
                print(
                    'For the label {},  drop rate {},  round {}, epoch {}.'.format(label_select, drop_rate, train_round,
                                                                                   epoch))
                print(
                    'The train acc is {}, the val acc is {}, the test acc is {}.'.format(train_acc, val_acc, test_acc))
                print('The train f1 is {}, the val f1 is {}, the test f1 is {}.'.format(train_f1, val_f1, test_f1))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_test_f1 = test_f1
            average_val_acc += best_val_acc
            average_test_acc += best_test_acc
            result_acc_list.append(best_test_acc)
            if best_test_acc > best_test_acc_under_best_val:
                best_test_acc_under_best_val = best_test_acc
            average_val_f1 += best_val_f1
            average_test_f1 += best_test_f1
            result_f1_list.append(best_test_f1)
            if best_test_f1 > best_test_f1_under_best_val:
                best_test_f1_under_best_val = best_test_f1
        average_val_acc /= train_round_number
        average_test_acc /= train_round_number
        test_acc_std = float(np.std(result_acc_list))
        average_val_f1 /= train_round_number
        average_test_f1 /= train_round_number
        test_f1_std = float(np.std(result_f1_list))
        result_statistic.loc[result_statistic.shape[0]] = {'dataset': train_dataset,
                                                           'date': '{}_{}'.format(args.month, args.date_day),
                                                           'label': label_select, 'backbone': backbone,
                                                           'drop_method': drop_method, 'drop_rate': drop_rate,
                                                           'average_val_acc': round(average_val_acc, 4),
                                                           'average_test_acc': round(average_test_acc, 4),
                                                           'test_acc_std': round(test_acc_std, 4),
                                                           'best_test_acc_under_best_val': round(
                                                               best_test_acc_under_best_val, 4),
                                                           'average_val_f1': round(average_val_f1, 4),
                                                           'average_test_f1': round(average_test_f1, 4),
                                                           'test_f1_std': round(test_acc_std, 4),
                                                           'best_test_f1_under_best_val':
                                                               round(best_test_f1_under_best_val, 4)}

save_path = os.path.join('..', '..', 'result', train_dataset, '{}_{}'.format(args.month, args.date_day))
if not os.path.exists(save_path):
    os.makedirs(save_path)
result_statistic.to_excel(os.path.join(save_path, '{}_{}_{}.xlsx'.format(backbone, drop_method, file_id)))
print('Mission complete.')
