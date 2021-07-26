import os
import sys
import torch
import pandas as pd
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import backbone_version.model as Md


def get_link_labels(pos_edge_index: Tensor, neg_edge_index: Tensor):
    num_edges = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_edges, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


cuda_device = 0

train_dataset = 'Cora'  # ['Cora', 'CiteSeer', 'PubMed']:
drop_method = 'DropMessage' # ['DropNode', 'DropEdge', 'Dropout', 'DropMessage']:
drop_rate = 0  # [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
backbone = 'APPNP'  # ['GAT', 'GCN', 'APPNP']:
unbias = True

# random generate train, validate, test mask
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="../TestDataset/{}".format(train_dataset), name=train_dataset, transform=T.NormalizeFeatures())
# preprocess for training: Train:Val:Test=0.85:0.05:0.1
data = train_test_split_edges(dataset[0])
data = data.to(device=device)

# Model
class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.backbone = backbone
        self.gnn1 = Md.OurModelLayer(feature_num, 64, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)
        self.gnn2 = Md.OurModelLayer(64, output_num, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)

    def encode(self, x: Tensor, pos_edge_index: Adj, drop_rate: float = 0):
        x = self.gnn1(x, pos_edge_index, drop_rate)
        if self.backbone == 'GAT':
            x = F.elu(x)
        else:
            x = F.relu(x)
        x = self.gnn2(x, pos_edge_index, drop_rate)
        return x

    def decode(self, z: Tensor, pos_edge_index: Adj, neg_egde_index: Adj):
        edge_index = torch.cat([pos_edge_index, neg_egde_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z: Tensor):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


unbias_rate = drop_rate if unbias else 0
model = Model(dataset.num_features, dataset.num_classes, backbone, drop_method, unbias_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train(data):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )

    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index, drop_rate)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    link_labels = link_labels.to(device)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(data):
    model.eval()
    perfs = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        z = model.encode(data.x, data.train_pos_edge_index)
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))

    return perfs


best_val_perf = test_perf = 0
epoch_num = 500
for epoch in range(epoch_num):
    train_loss = train(data)
    val_perf, tmp_test_perf = test(data)

    log = 'Epoch: {:03d}, Loss: {:.6f}, Val acc: {:.6f}, Test acc: {:.6f}'
    print(log.format(epoch, train_loss, val_perf, tmp_test_perf))

    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf

# z = model.encode(data.x, data.train_pos_edge_index)
# final_edge_index = model.decode_all(z)

print('Mission completes.')
print('The best val acc is {}, and the test acc is {}.'.format(best_val_perf, test_perf))

