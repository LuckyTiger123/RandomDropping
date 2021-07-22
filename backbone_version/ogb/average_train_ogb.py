import os
import sys
import torch
import argparse
import gc
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=10)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-d', '--dataset', type=str, default='ogbn-arxiv')
parser.add_argument('-e', '--epoch', type=int, default=500)
parser.add_argument('-bb', '--backbone', type=str, default='GCN')
parser.add_argument('-dm', '--drop_method', type=str, default='Dropout')
parser.add_argument('-hs', '--heads', type=int, default=1)
parser.add_argument('-k', '--K', type=int, default=10)
parser.add_argument('-a', '--alpha', type=float, default=0.1)
parser.add_argument('-hd', '--hidden_dimensions', type=int, default=256)
parser.add_argument('-nl', '--num_layers', type=int, default=3)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

train_dataset = args.dataset
backbone = args.backbone
drop_method = args.drop_method
num_layers = args.num_layers

cuda_device = args.cuda
heads = args.heads
K = args.K
alpha = args.alpha
hidden_dimensions = args.hidden_dimensions
file_id = args.file_id

train_round_number = args.train_round
epoch_num = args.epoch

drop_rate_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
unbias_list = [True, False]

# random generate train, validate, test mask
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

result_statistic = pd.DataFrame(
    columns=['dataset', 'backbone', 'drop_method', 'drop_rate', 'unbias', 'best_test_acc', 'best_val_acc',
             'average_val_acc', 'average_test_acc', 'best_test_acc_under_best_val'])

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = PygNodePropPredDataset(name=train_dataset, root='/home/luckytiger/TestDataset', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
evaluator = Evaluator(name=train_dataset)


# Model
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            Md.OurModelLayer(in_channels, hidden_channels, drop_method, backbone, unbias=unbias, heads=heads, K=K,
                             alpha=alpha))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                Md.OurModelLayer(hidden_channels, hidden_channels, drop_method, backbone, unbias=unbias, heads=heads,
                                 K=K, alpha=alpha))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            Md.OurModelLayer(hidden_channels, out_channels, drop_method, backbone, unbias=unbias, heads=heads, K=K,
                             alpha=alpha))

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, drop_rate)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t, drop_rate)
    loss = F.cross_entropy(out[train_idx], data.y.squeeze(1)[train_idx])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


for drop_rate in drop_rate_list:
    for unbias in unbias_list:
        best_test_acc = best_val_acc = average_val_acc = average_test_acc = best_test_acc_under_best_val = 0
        if unbias:
            unbias_rate = drop_rate
        else:
            unbias_rate = 0
        gc.collect()
        model = Model(data.num_features, hidden_dimensions, dataset.num_classes, num_layers, backbone, drop_method,
                      unbias_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
        for train_round in range(train_round_number):
            gc.collect()
            model.reset_parameters()
            best_val = best_test = 0
            for epoch in range(epoch_num):
                train()
                train_acc, val_acc, current_test_acc = test()
                print('---------------------------------------------------------------------------')
                print('For the drop rate {}, unbias {}, round {}.'.format(drop_rate, unbias, train_round))
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

        average_val_acc /= train_round_number
        average_test_acc /= train_round_number

        result_statistic.loc[result_statistic.shape[0]] = {'dataset': train_dataset, 'backbone': backbone,
                                                           'drop_method': drop_method, 'drop_rate': drop_rate,
                                                           'unbias': unbias, 'best_test_acc': round(best_test_acc, 4),
                                                           'best_val_acc': round(best_val_acc, 4),
                                                           'average_val_acc': round(average_val_acc, 4),
                                                           'average_test_acc': round(average_test_acc, 4),
                                                           'best_test_acc_under_best_val': round(
                                                               best_test_acc_under_best_val, 4)}

save_path = os.path.join('..', '..', 'result', train_dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)
result_statistic.to_excel(os.path.join(save_path, '{}_{}_{}_bn.xlsx'.format(backbone, drop_method, file_id)))
print('Mission complete.')
