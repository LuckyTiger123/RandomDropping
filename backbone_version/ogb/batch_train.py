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
from torch_geometric.data import NeighborSampler

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

train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[-1], batch_size=1024, shuffle=True,
                               num_workers=12)
subgraph_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False,
                                  num_workers=12)

for batch_size, n_id, adjs in train_loader:
    pass
