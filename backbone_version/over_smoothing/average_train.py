import os
import gc
import sys
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch_geometric.utils as utils

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md
import backbone_version.over_smoothing.MADGap as MAD

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', type=int, default=0)
parser.add_argument('-t', '--train_round', type=int, default=20)
parser.add_argument('-f', '--file_id', type=int, default=0)
parser.add_argument('-d', '--dataset', type=str, default='Cora')
parser.add_argument('-e', '--epoch', type=int, default=200)
parser.add_argument('-bb', '--backbone', type=str, default='GCN')
parser.add_argument('-ln', '--layer_num', type=int, default=1)
parser.add_argument('-hd', '--hidden_dimension', type=int, default=16)
parser.add_argument('-r', '--rand_seed', type=int, default=0)
args = parser.parse_args()

train_dataset = args.dataset
backbone = args.backbone
num_layers = args.layer_num
cuda_device = args.cuda
hidden_dimension = args.hidden_dimension
file_id = args.file_id
train_round_number = args.train_round
epoch_num = args.epoch

drop_rate_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
drop_method_list = ['Dropout', 'DropEdge', 'DropNode', 'DropMessage']

# random seed fix
random_seed = args.rand_seed
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

result_statistic = pd.DataFrame(
    columns=['dataset', 'backbone', 'drop_method', 'drop_rate', 'unbias', 'best_test_acc', 'average_test_acc',
             'best_MAD_gap', 'average_MAD_gap'])

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/{}".format(train_dataset), name=train_dataset,
                    transform=T.NormalizeFeatures())

data = dataset[0].to(device=device)

dense_edge_index = utils.to_dense_adj(data.edge_index).squeeze()
three_neighborhood = dense_edge_index
k_neighborhood = dense_edge_index
ks_neighborhood = dense_edge_index
for i in range(2):
    k_neighborhood = k_neighborhood.matmul(dense_edge_index)
    three_neighborhood += k_neighborhood
    ks_neighborhood += k_neighborhood

three_hop_edge_index = utils.dense_to_sparse(three_neighborhood)
neb_edge_index = three_hop_edge_index[0]

for i in range(4):
    k_neighborhood = k_neighborhood.matmul(dense_edge_index)
    ks_neighborhood += k_neighborhood

rmt_edge_index = utils.dense_to_sparse(
    torch.ones([data.num_nodes, data.num_nodes]).to(data.edge_index.device) - utils.to_dense_adj(
        ks_neighborhood.nonzero().t()))[0]


class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(
                Md.OurModelLayer(in_channels, out_channels, drop_method, backbone, unbias=unbias))
        else:
            self.convs.append(
                Md.OurModelLayer(in_channels, hidden_channels, drop_method, backbone, unbias=unbias))
            for _ in range(num_layers - 2):
                self.convs.append(
                    Md.OurModelLayer(hidden_channels, hidden_channels, drop_method, backbone, unbias=unbias))
            self.convs.append(
                Md.OurModelLayer(hidden_channels, out_channels, drop_method, backbone, unbias=unbias))

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.num_layers == 1:
            return self.convs[0](x, edge_index, drop_rate)

        for _, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, drop_rate)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, drop_rate)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, drop_rate)
    _, pred = out.max(dim=1)
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    validate_acc = validate_correct / int(data.val_mask.sum())
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())
    return train_acc, validate_acc, test_acc, out


cal = MAD.MADGap_calculator()

for drop_rate in drop_rate_list:
    for drop_method in drop_method_list:
        gc.collect()
        if drop_method == 'DropNode':
            unbias = drop_rate
        else:
            unbias = 0
        best_test_acc = average_test_acc = best_MAD_gap = average_MAD_gap = 0
        model = Model(dataset.num_features, hidden_dimension, dataset.num_classes, num_layers, backbone, drop_method,
                      unbias).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
        for train_round in range(train_round_number):
            model.reset_parameters()
            best_val = best_test = 0
            f_R = None
            for epoch in range(epoch_num):
                train()
                train_acc, val_acc, current_test_acc, c_R = test()
                print('---------------------------------------------------------------------------')
                print('For the {}.'.format(drop_method))
                print('For the drop rate {}, unbias {}, round {}.'.format(drop_rate, unbias, train_round))
                print('For the {} epoch, the train acc is {}, '
                      'the val acc is {}, the test acc is {}.'.format(epoch,
                                                                      train_acc,
                                                                      val_acc,
                                                                      current_test_acc))
                if val_acc > best_val:
                    best_val = val_acc
                    best_test = current_test_acc
                    f_R = c_R
            average_test_acc += best_test
            if best_test > best_test_acc:
                best_test_acc = best_test
            mad_gap = cal(f_R, neb_edge_index, rmt_edge_index)
            mad_gap = float(mad_gap)
            average_MAD_gap += mad_gap
            if mad_gap > best_MAD_gap:
                best_MAD_gap = mad_gap

        average_MAD_gap /= train_round_number
        average_test_acc /= train_round_number
        result_statistic.loc[result_statistic.shape[0]] = {'dataset': train_dataset, 'backbone': backbone,
                                                           'drop_method': drop_method, 'drop_rate': drop_rate,
                                                           'unbias': unbias, 'best_test_acc': round(best_test_acc, 4),
                                                           'average_test_acc': round(average_test_acc, 4),
                                                           'best_MAD_gap': round(best_MAD_gap, 4),
                                                           'average_MAD_gap': round(average_MAD_gap, 4)}

save_path = os.path.join('..', '..', 'result', 'over_smoothing', train_dataset)
if not os.path.exists(save_path):
    os.makedirs(save_path)
result_statistic.to_excel(os.path.join(save_path, '{}_{}.xlsx'.format(num_layers, file_id)))
