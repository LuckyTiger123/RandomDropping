import os
import gc
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch.nn.functional as F
import torch_geometric.transforms as T

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.drop_rate.drop_model as Md

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=20)
parser.add_argument('-f', '--file_id', type=int, default=1)
parser.add_argument('-e', '--epoch', type=int, default=1000)
parser.add_argument('-d', '--dataset', type=str, default='Cora')
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

# configuration parameters
train_dataset = args.dataset
cuda_device = args.cuda
train_round_number = args.train_round
epoch_num = args.epoch
file_id = args.file_id

drop_rate_list = [-1, 2]

# fix random seed
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

result_statistic = pd.DataFrame(
    columns=['dataset', 'drop_rate', 'best_test_acc', 'best_val_acc', 'average_val_acc', 'average_test_acc', 'test_std',
             'best_test_acc_under_best_val'])

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/{}".format(train_dataset), name=train_dataset,
                    transform=T.NormalizeFeatures())

data = dataset[0].to(device=device)


# Model
class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num, drop_method):
        super(Model, self).__init__()
        self.gnn1 = Md.ModifiedGCN(feature_num, 16, drop_method=drop_method)
        self.gnn2 = Md.ModifiedGCN(16, output_num, drop_method=drop_method)
        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj):
        x = self.gnn1(x, edge_index)
        x = F.relu(x)
        x = self.gnn2(x, edge_index)
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    _, pred = out.max(dim=1)
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    validate_acc = validate_correct / int(data.val_mask.sum())
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())
    return train_acc, validate_acc, test_acc


for drop_rate in drop_rate_list:
    gc.collect()
    best_test_acc = best_val_acc = average_val_acc = average_test_acc = best_test_acc_under_best_val = 0
    result_list = []
    model = Model(dataset.num_features, dataset.num_classes, drop_method=drop_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
    for train_round in range(train_round_number):
        model.reset_parameters()
        best_val = best_test = 0
        for epoch in range(epoch_num):
            train()
            train_acc, val_acc, current_test_acc = test()
            print('---------------------------------------------------------------------------')
            print('For the drop rate {}, round {}.'.format(drop_rate, train_round))
            print('For the {} epoch, the train acc is {}, '
                  'the val acc is {}, the test acc is {}.'.format(epoch,
                                                                  train_acc,
                                                                  val_acc,
                                                                  current_test_acc))
            if current_test_acc > best_test_acc:
                best_test_acc = current_test_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if val_acc > best_val:
                best_val = val_acc
                best_test = current_test_acc
                if current_test_acc > best_test_acc_under_best_val:
                    best_test_acc_under_best_val = current_test_acc
        average_val_acc += best_val
        average_test_acc += best_test
        result_list.append(best_test)

    average_val_acc /= train_round_number
    average_test_acc /= train_round_number
    test_std = float(np.std(result_list))

    result_statistic.loc[result_statistic.shape[0]] = {'dataset': train_dataset, 'drop_rate': drop_rate,
                                                       'best_test_acc': round(best_test_acc, 4),
                                                       'best_val_acc': round(best_val_acc, 4),
                                                       'average_val_acc': round(average_val_acc, 4),
                                                       'average_test_acc': round(average_test_acc, 4),
                                                       'test_std': round(test_std, 4),
                                                       'best_test_acc_under_best_val': round(
                                                           best_test_acc_under_best_val, 4)}

save_path = os.path.join('..', '..', 'result', 'drop_rate', train_dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)
result_statistic.to_excel(os.path.join(save_path, '{}.xlsx'.format(file_id)))
print('Mission complete.')
