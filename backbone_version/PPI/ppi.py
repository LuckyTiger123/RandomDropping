import os
import gc
import sys
import torch
import random
from torch import Tensor
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.typing import Adj
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from backbone_version.PPI.modified_GAT import ModifiedGATConv
from backbone_version.model import DropBlock

cuda_device = 6

train_dataset = 'PPI'
drop_method = 'DropEdge'
drop_rate = 0.1
backbone = 'GCN'

# random generate train, validate, test mask
random_seed = 1
random.seed(1)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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


model = Model(dataset_train.num_features, dataset_train.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_op = torch.nn.BCEWithLogitsLoss()
epoch_num = 100


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


best_val_f1 = best_test_f1 = 0
for epoch in range(epoch_num):
    gc.collect()
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)

    print('For the {} epoch, the train loss is {}, the val f1 is {}, the test f1 is {}.'.format(epoch, loss,
                                                                                                val_f1,
                                                                                                test_f1))
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_test_f1 = test_f1

print('Mission completes.')
print('The best val acc is {}, and the test acc is {}.'.format(best_val_f1, best_test_f1))
