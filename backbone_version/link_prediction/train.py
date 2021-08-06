import os
import sys
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

train_dataset = 'Cora'
cuda_device = 3
drop_method = 'Dropout'
drop_rate = 0.1
backbone = 'GAT'
unbias = False

# random generate train, validate, test mask
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')
# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/{}".format(train_dataset), name=train_dataset,
                    transform=T.NormalizeFeatures())

data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
print(data)


class Net(torch.nn.Module):
    def __init__(self, feature_num, output_num, backbone, drop_method, unbias):
        super(Net, self).__init__()
        self.backbone = backbone
        self.gnn1 = Md.OurModelLayer(feature_num, 128, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)
        self.gnn2 = Md.OurModelLayer(128, output_num, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)

    def encode(self, x, edge_index, drop_rate: float = 0):
        x = self.gnn1(x, edge_index, drop_rate)
        x = x.relu()
        return self.gnn2(x, edge_index, drop_rate)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


if unbias:
    unbias_rate = drop_rate
else:
    unbias_rate = 0

model = Net(dataset.num_features, 64, backbone, drop_method, unbias_rate).to(device)
# model = Net(dataset.num_features, 64).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(data):
    model.train()

    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test(data):
    model.eval()

    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results


best_val_auc = test_auc = 0
for epoch in range(1, 101):
    loss = train(data)
    val_auc, tmp_test_auc = test(data)
    if val_auc > best_val_auc:
        best_val = val_auc
        test_auc = tmp_test_auc
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

print('The best val auc is {}, the best test auc is {}.'.format(best_val_auc, test_auc))
z = model.encode(data.x, data.train_pos_edge_index)
final_edge_index = model.decode_all(z)
